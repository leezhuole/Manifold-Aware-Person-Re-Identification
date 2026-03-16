from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import geoopt
from functools import partial

from ..loss.triplet import euclidean_dist, finsler_drift_dist

from .pooling import GeneralizedMeanPoolingP
from .vit import vit_base_patch16_224_TransReID
from .mobilenetv2 import mobilenetv2_x1_4
from geoopt import PoincareBall


# class HyperbolicClassifier(nn.Module):
#     """Classify by negative hyperbolic distance to learnable prototypes."""

#     def __init__(self, in_features, out_features, manifold):
#         super().__init__()
#         self.manifold = manifold
#         self.in_features = in_features
#         self.out_features = out_features
#         self.prototypes = geoopt.ManifoldParameter(
#             data=self.manifold.random(out_features, in_features),
#             manifold=self.manifold,
#         )

#     def forward(self, x):
#         # logits = -d_M(x, prototype_k)
#         dists = self.manifold.dist(
#             x.unsqueeze(1),
#             self.prototypes.unsqueeze(0),
#         )
#         return -dists


class resnet50(nn.Module):
    """
    Adapted such that the model's forward pass outputs embeddings that are
    on the defined manifold (if provided). Example: Poincare ball. 

    """
    def __init__(self, num_classes=0, pretrained=True, manifold=None):
        super(resnet50, self).__init__()
        self.num_classes = num_classes
        self.manifold = manifold
        self.dist_func = euclidean_dist
        self.embedding_dim = 2048
        self.memory_bank_dim = 2048
        self.supports_manifold = True
        
        # resnet50 backbone
        resnet = torchvision.models.resnet50(pretrained=pretrained)

        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.base = nn.Sequential(
            resnet.conv1, 
            resnet.bn1, 
            resnet.relu, 
            resnet.maxpool,
            resnet.layer1, 
            nn.InstanceNorm2d(256),
            resnet.layer2, 
            nn.InstanceNorm2d(512),
            resnet.layer3, 
            nn.InstanceNorm2d(1024),
            resnet.layer4
        )

        # pooling
        self.pool = GeneralizedMeanPoolingP(output_size=(1, 1))

        # bnneck and classifier
        self.bn_neck = nn.BatchNorm1d(2048)
        init.constant_(self.bn_neck.weight, 1)
        init.constant_(self.bn_neck.bias, 0)
        self.bn_neck.bias.requires_grad_(False)
        if self.num_classes > 0:
            # if self.manifold is not None:
            #     self.classifier = HyperbolicClassifier(2048, self.num_classes, self.manifold)
            # else:
            self.classifier = nn.Linear(2048, self.num_classes, bias=False)

    def forward(self, x):
        x = self.base(x)              # Euclidean features
        emb = self.pool(x)
        emb = emb.view(x.size(0), -1)
        f = self.bn_neck(emb)
        f_norm = F.normalize(f)

        if self.manifold is not None:
            # Follow the pattern used in geoopt/geoopt (vision resnet examples) and
            # facebookresearch/poincare-embeddings: keep Euclidean processing in the
            # tangent space, then move the embedding onto the manifold at the end.
            f_hyp = self.manifold.expmap0(f_norm, dim=-1)
            f_hyp = geoopt.ManifoldTensor(f_hyp, manifold=self.manifold)
            emb_hyp = self.manifold.expmap0(emb, dim=-1)
            emb_hyp = geoopt.ManifoldTensor(emb_hyp, manifold=self.manifold)
        else:
            f_hyp = f_norm
            emb_hyp = emb

        if self.training:
            logits = None
            if self.num_classes > 0:
                # if self.manifold is not None:
                #     logits = self.classifier(f_hyp)
                # else:
                logits = self.classifier(f)
            if logits is not None:
                return emb_hyp, f_hyp, logits
            else:
                raise ValueError("Number of classes must be greater than 0 to compute logits.")
        else:
            return f_hyp


class FinslerDriftHead(nn.Module):
    def __init__(self, input_dim, output_dim, max_norm=0.95):
        super(FinslerDriftHead, self).__init__()
        hidden_dim = max(1, input_dim // 2)
        self.max_norm = max_norm
        # Bias set to False to ensure f(0) = 0
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

        # Initialize last layer to near-zero
        init.normal_(self.block[-1].weight, std=0.001)
        # init.constant_(self.block[-1].bias, 0)    # No bias to initialize

    def forward(self, x):
        drift = self.block(x)
        return scale_drift_vector(drift, max_norm=self.max_norm)


def scale_drift_vector(drift, max_norm=0.95, eps=1e-6):
    norms = torch.norm(drift, p=2, dim=1, keepdim=True)

    # Option 1: Simple Norm Clipping + Scaling
    # scaled = torch.clamp(norms, max=max_norm)
    # drift = drift * (scaled / (norms + eps))

    # Option 2: Soft Tanh-Scaling
    # scale = max_norm * torch.tanh(norms / max_norm) / (norms + eps)
    # drift = drift * scale

    # Option 3: Sigmoid Gated Scaling
    scaling_factor = torch.sigmoid(norms)
    unit_drift = drift / (norms + eps)
    drift = unit_drift * (scaling_factor * max_norm)

    # Option 4: Log-Barrier / Soft-Maximum
    # beta = 5.0
    # max_norm_t = torch.tensor(max_norm, device=norms.device)
    # smooth_norm = - (1 / beta) * torch.logaddexp(-beta * norms, -beta * max_norm_t)
    # drift = drift * (smooth_norm / (norms + eps))
    return drift


class DomainConditionedDriftHead(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_domains,
        context_dim=64,
        max_norm=0.95,
        residual_scale=0.1,
    ):
        super(DomainConditionedDriftHead, self).__init__()
        if num_domains <= 0:
            raise ValueError("num_domains must be positive for DomainConditionedDriftHead")

        hidden_dim = max(1, input_dim // 2)
        self.num_domains = int(num_domains)
        self.max_norm = max_norm
        self.residual_scale = float(residual_scale)

        self.domain_embeddings = nn.Embedding(self.num_domains, context_dim)
        self.domain_projection = nn.Linear(context_dim, output_dim, bias=False)
        self.domain_gate = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.Sigmoid(),
        )
        self.residual_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

        init.normal_(self.domain_embeddings.weight, std=0.01)
        init.normal_(self.domain_projection.weight, std=0.01)
        init.normal_(self.residual_block[-1].weight, std=0.001)

    def _resolve_domain_context(self, domain_ids=None, domain_probs=None):
        if domain_probs is not None:
            if domain_probs.dim() != 2 or domain_probs.size(1) != self.num_domains:
                raise ValueError("domain_probs must have shape (batch, num_domains)")
            return domain_probs @ self.domain_embeddings.weight

        if domain_ids is None:
            raise ValueError("Either domain_ids or domain_probs must be provided for domain-conditioned drift")

        return self.domain_embeddings(domain_ids)

    def forward(self, x, domain_ids=None, domain_probs=None):
        domain_context = self._resolve_domain_context(domain_ids=domain_ids, domain_probs=domain_probs)
        domain_drift = self.domain_projection(domain_context)
        gated_domain_drift = domain_drift * self.domain_gate(x)
        residual_drift = self.residual_block(x)
        drift = gated_domain_drift + self.residual_scale * residual_drift
        return scale_drift_vector(drift, max_norm=self.max_norm)


class resnet50_finsler(nn.Module):
    """
    ResNet50 with a learned drift vector branch for Finsler/Randers distance.
    """
    def __init__(
        self,
        num_classes=0,
        pretrained=True,
        manifold=None,
        use_drift_in_eval=True,
        memory_bank_mode="full",
        drift_dim=2048,
        drift_method="symmetric_trapezoidal",
        drift_conditioning="instance",
        num_domains=0,
        domain_embed_dim=64,
        infer_domain_conditioning=False,
        domain_temperature=1.0,
        domain_residual_scale=0.1,
    ):
        super(resnet50_finsler, self).__init__()
        self.num_classes = num_classes
        self.manifold = manifold
        self.identity_dim = 2048
        self.dist_func = partial(finsler_drift_dist, identity_dim=self.identity_dim, method=drift_method)
        if drift_dim is None:
            drift_dim = self.identity_dim
        if int(drift_dim) <= 0:
            raise ValueError("drift_dim must be a positive integer")
        self.drift_dim = int(drift_dim)
        if drift_conditioning not in {"instance", "domain"}:
            raise ValueError("drift_conditioning must be either 'instance' or 'domain'")
        self.drift_conditioning = drift_conditioning
        self.num_domains = int(num_domains)
        self.domain_temperature = float(domain_temperature)
        self.infer_domain_conditioning = bool(infer_domain_conditioning)
        self.embedding_dim = self.identity_dim + self.drift_dim
        self.use_drift_in_eval = bool(use_drift_in_eval)
        self.supports_manifold = False

        if self.drift_conditioning == "domain" and self.num_domains <= 0:
            raise ValueError("num_domains must be positive when drift_conditioning='domain'")
        if self.domain_temperature <= 0:
            raise ValueError("domain_temperature must be positive")

        if memory_bank_mode not in {"full", "identity"}:
            raise ValueError("memory_bank_mode must be 'full' or 'identity'")
        self.memory_bank_dim = self.embedding_dim if memory_bank_mode == "full" else self.identity_dim

        resnet = torchvision.models.resnet50(pretrained=pretrained)

        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.base = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            nn.InstanceNorm2d(256),
            resnet.layer2,
            nn.InstanceNorm2d(512),
            resnet.layer3,
            nn.InstanceNorm2d(1024),
            resnet.layer4,
        )

        self.pool = GeneralizedMeanPoolingP(output_size=(1, 1))

        self.bn_neck = nn.BatchNorm1d(self.identity_dim)
        init.constant_(self.bn_neck.weight, 1)
        init.constant_(self.bn_neck.bias, 0)
        self.bn_neck.bias.requires_grad_(False)

        if self.drift_conditioning == "domain":
            self.drift_head = DomainConditionedDriftHead(
                self.identity_dim,
                self.drift_dim,
                num_domains=self.num_domains,
                context_dim=domain_embed_dim,
                residual_scale=domain_residual_scale,
            )
            self.domain_classifier = nn.Linear(self.identity_dim, self.num_domains)
            init.normal_(self.domain_classifier.weight, std=0.001)
            init.constant_(self.domain_classifier.bias, 0)
        else:
            self.drift_head = FinslerDriftHead(self.identity_dim, self.drift_dim)
            self.domain_classifier = None

        if self.num_classes > 0:
            self.classifier = nn.Linear(self.identity_dim, self.num_classes, bias=False)

    def select_eval_embedding(self, embedding):
        if self.use_drift_in_eval:
            return embedding
        return embedding[:, :self.identity_dim]

    def _predict_domain_logits(self, identity):
        if self.domain_classifier is None:
            return None
        return self.domain_classifier(identity.detach())

    def _resolve_domain_probs(self, identity, domain_ids=None, domain_probs=None):
        domain_logits = self._predict_domain_logits(identity)
        if domain_probs is not None:
            return domain_probs, domain_logits

        if domain_ids is not None:
            resolved = F.one_hot(domain_ids, num_classes=self.num_domains).to(dtype=identity.dtype)
            return resolved, domain_logits

        if self.infer_domain_conditioning and domain_logits is not None:
            return torch.softmax(domain_logits / self.domain_temperature, dim=1), domain_logits

        uniform_probs = identity.new_full((identity.size(0), self.num_domains), 1.0 / self.num_domains)
        return uniform_probs, domain_logits

    def forward(self, x, domain_ids=None, domain_probs=None, return_aux=False):
        x = self.base(x)
        emb = self.pool(x)
        emb = emb.view(x.size(0), -1)
        identity = self.bn_neck(emb)
        identity_norm = F.normalize(identity) # L2-normalized identity vector
        aux_outputs = None

        if self.drift_conditioning == "domain":
            resolved_domain_probs, domain_logits = self._resolve_domain_probs(
                identity,
                domain_ids=domain_ids,
                domain_probs=domain_probs,
            )
            drift = self.drift_head(emb, domain_probs=resolved_domain_probs)
            aux_outputs = {
                "domain_logits": domain_logits,
                "domain_probs": resolved_domain_probs,
            }
        else:
            drift = self.drift_head(emb)
        
        # -- Enforcing orthoganality between identity and drift --
        inner_product = torch.sum(drift * identity_norm, dim=1, keepdim=True)
        drift = drift - inner_product * identity_norm
        
        combined_emb = torch.cat([emb, drift], dim=1)
        combined_f = torch.cat([identity_norm, drift], dim=1)

        if self.training:
            logits = None
            if self.num_classes > 0:
                logits = self.classifier(identity)
            if logits is not None:
                if return_aux or aux_outputs is not None:
                    return combined_emb, combined_f, logits, aux_outputs
                return combined_emb, combined_f, logits
            raise ValueError("Number of classes must be greater than 0 to compute logits.")
        output = self.select_eval_embedding(combined_f)
        if return_aux and aux_outputs is not None:
            return output, aux_outputs
        return output



class mobilenetv2(nn.Module):
    """
    Adapted such that the model's forward pass outputs embeddings that are
    on the defined manifold (if provided). Example: Poincare ball.

    """
    def __init__(self, num_classes=0, pretrained=True, manifold=None):
        super(mobilenetv2, self).__init__()
        self.num_classes = num_classes
        self.manifold = manifold
        self.dist_func = euclidean_dist
        self.embedding_dim = 1792
        self.memory_bank_dim = 1792
        self.supports_manifold = True
        
        # mobilenetv2 backbone
        mobilenet = mobilenetv2_x1_4(num_classes=num_classes)

        if pretrained:
            model_path = './checkpoints/mobilenetv2_1.4-bc1cc36b.pth' # your pre-trained weight path here
            pretrain_dict = torch.load(model_path)
            model_dict = mobilenet.state_dict()
            pretrain_dict = {
                k: v
                for k, v in pretrain_dict.items()
                if k in model_dict and model_dict[k].size() == v.size()
            }
            model_dict.update(pretrain_dict)
            mobilenet.load_state_dict(model_dict)
        
        self.base = mobilenet.base

        # pooling
        self.pool = GeneralizedMeanPoolingP(output_size=(1, 1))

        # cnn backbone
        self.bn_neck = nn.BatchNorm1d(1792)
        init.constant_(self.bn_neck.weight, 1)
        init.constant_(self.bn_neck.bias, 0)
        self.bn_neck.bias.requires_grad_(False)
        if self.num_classes > 0:
                self.classifier = nn.Linear(1792, self.num_classes, bias=False)

    def forward(self, x):
        x = self.base(x)
        emb = self.pool(x)
        emb = emb.view(x.size(0), -1)
        f = self.bn_neck(emb)
        if self.training:
            logits = self.classifier(f)
            return emb, F.normalize(f), logits
        else:
            return F.normalize(f)


class vit_base_patch16(nn.Module):
    """
    Adapted such that the model's forward pass outputs embeddings that are
    on the defined manifold (if provided). Example: Poincare ball.

    """
    def __init__(self, num_classes=0, pretrained=True, manifold=None):
        super(vit_base_patch16, self).__init__()
        last_stride = 1
        self.neck = 'bnneck'
        self.neck_feat = 'after'
        self.in_planes = 768
        self.num_classes = num_classes
        self.manifold = manifold        
        self.dist_func = euclidean_dist
        self.embedding_dim = 768
        self.memory_bank_dim = 768
        self.supports_manifold = True

        # vit backbone
        self.base = vit_base_patch16_224_TransReID(camera=0, view=0, local_feature=False)
        
        if pretrained:
            model_path = './checkpoints/jx_vit_base_p16_224-80ecf9dd.pth' # your pre-trained weight path here
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        # bnneck and classifier
        self.bn_neck = nn.BatchNorm1d(768)
        init.constant_(self.bn_neck.weight, 1)
        init.constant_(self.bn_neck.bias, 0)
        self.bn_neck.bias.requires_grad_(False)
        if self.num_classes > 0:
                self.classifier = nn.Linear(768, self.num_classes, bias=False)

    def forward(self, x):
        cls_token = self.base(x)  # class token
        f = self.bn_neck(cls_token)  # bnneck

        if self.training:
            logits = self.classifier(f)
            return cls_token, torch.nn.functional.normalize(f), logits
        else:
            return torch.nn.functional.normalize(f)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
