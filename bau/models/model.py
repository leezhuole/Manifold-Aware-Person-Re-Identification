from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import geoopt

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
        
        # resnet50 backbone
        resnet = torchvision.models.resnet50(pretrained=pretrained)

        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, nn.InstanceNorm2d(256),
            resnet.layer2, nn.InstanceNorm2d(512),
            resnet.layer3, nn.InstanceNorm2d(1024),
            resnet.layer4, GeneralizedMeanPoolingP(output_size=(1, 1))
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



class mobilenetv2(nn.Module):
    """
    Adapted such that the model's forward pass outputs embeddings that are
    on the defined manifold (if provided). Example: Poincare ball.

    """
    def __init__(self, num_classes=0, pretrained=True, manifold=None):
        super(mobilenetv2, self).__init__()
        self.num_classes = num_classes
        self.manifold = manifold
        
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
