  
from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .evaluation_metrics import accuracy
from .loss import CrossEntropyLabelSmooth, TripletLoss

from .utils.meters import AverageMeter
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from .loss.triplet import euclidean_dist, finsler_drift_dist

# optional wandb logging (kept minimal and non-intrusive)
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover - wandb is optional
    wandb = None


class BAUTrainer(object):
    def __init__(self, model, memory_bank, num_classes, margin, lam=1.5, k=10, manifold=None, manifold_chunk_size=None,
                 use_aug_ce=False, use_align=True, use_drift_align=False, use_uniform=True, use_domain=True, use_triplet=True, 
                 use_ce=True, alpha=None, alpha_module=None, bidirectional_triplet=False,
                 use_omega_reg=False, omega_reg_weight=1.0, domain_token_loss_weight=0.0):
        
        """
        Note that static alpha is not used during training. It is only set as a default when no alpha_module is provided
        
        """
        super(BAUTrainer, self).__init__()
        self.model = model
        self.memory_bank = memory_bank
        module = getattr(model, 'module', model)
        self.dist_func = getattr(module, 'dist_func', euclidean_dist)
        self.identity_dim = getattr(module, 'identity_dim', None)  # e.g. 2048 for resnet50_finsler
        
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes=num_classes).cuda()
        self.criterion_domain_token = nn.CrossEntropyLoss().cuda()
        static_alpha = None if alpha_module is not None else alpha
        self.criterion_tri = TripletLoss(
            margin=margin,
            manifold=manifold,
            alpha=static_alpha,
            dist_func=self.dist_func,
            bidirectional=bidirectional_triplet,
        ).cuda()
        self.scaler = GradScaler()

        self.lam = lam
        self.k = k
        self.manifold = manifold
        self.manifold_chunk_size = manifold_chunk_size
        
        self.use_aug_ce = use_aug_ce
        self.use_align = use_align
        self.use_drift_align = use_drift_align
        self.use_uniform = use_uniform
        self.use_domain = use_domain
        self.use_triplet = use_triplet
        self.use_ce = use_ce
        self.use_omega_reg = use_omega_reg
        self.omega_reg_weight = omega_reg_weight
        self.domain_token_loss_weight = domain_token_loss_weight

        # Finsler space parameter
        self.alpha = alpha
        self.alpha_module = alpha_module

    def _get_alpha_value(self):
        if self.alpha_module is None:
            return self.alpha
        return self.alpha_module.value()

    def train(self, epoch, train_loader, optimizer, iters, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_align = AverageMeter()
        losses_drift_align = AverageMeter()
        losses_uniform = AverageMeter()
        losses_domain = AverageMeter()
        losses_domain_token = AverageMeter()
        losses_omega = AverageMeter()
        precisions = AverageMeter()

        time.sleep(1)
        end = time.time()
        for i in range(iters):
            batch_data = train_loader.next()
            # inputs_w: Weakly augmented images
            # inputs_s: Strongly augmented images
            inputs_w, inputs_s, pids, dids = self._parse_data(batch_data)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                # feedforward
                inputs = torch.cat([inputs_w, inputs_s], dim=0)
                domain_ids = torch.cat([dids, dids], dim=0)
                model_outputs = self.model(inputs, domain_ids=domain_ids)
                domain_token_logits = None
                if isinstance(model_outputs, tuple) and len(model_outputs) == 4:
                    emb, f, logits, aux_outputs = model_outputs
                    if isinstance(aux_outputs, dict):
                        domain_token_logits = aux_outputs.get("domain_logits")
                else:
                    emb, f, logits = model_outputs
                # f = F.normalize(f)        # already normalized in model
                emb_w, emb_s = emb.chunk(2)
                f_w, f_s = f.chunk(2)
                if self.identity_dim is not None and f_w.size(1) > self.identity_dim: # Check that there is a drift vector concatenated
                    f_w_align = f_w[:, :self.identity_dim]
                    f_s_align = f_s[:, :self.identity_dim]
                    f_w_drift = f_w[:, self.identity_dim:]
                    f_s_drift = f_s[:, self.identity_dim:]
                else:
                    f_w_align = f_w
                    f_s_align = f_s
                    f_w_drift = None
                    f_s_drift = None  
                logits_w, logits_s = logits.chunk(2)

                drift_norm_mean = None
                drift_norm_max = None
                module = getattr(self.model, 'module', self.model)
                drift_start = getattr(module, 'identity_dim', None)
                if drift_start is not None and f.size(1) > drift_start:
                    drift = f[:, drift_start:]
                    drft_norm = drift.norm(p=2, dim=1)
                    drift_norm_mean = drft_norm.mean()
                    drift_norm_max = drft_norm.max()

                # For Finsler models, create detached-identity versions of f_w/f_s
                # for domain and uniform losses.  This prevents repulsion gradients
                # from flowing into the backbone identity features, which would
                # conflict with alignment and cause identity embeddings to diverge.
                # if self.identity_dim is not None and f_w.size(1) > self.identity_dim:
                #     f_w_repulse = torch.cat([
                #         f_w[:, :self.identity_dim],
                #         f_w[:, self.identity_dim:]
                #     ], dim=1)
                #     f_s_repulse = torch.cat([
                #         f_s[:, :self.identity_dim].detach(),
                #         f_s[:, self.identity_dim:]
                #     ], dim=1)
                # else:
                #     f_w_repulse = f_w
                #     f_s_repulse = f_s

                # compute weights
                with torch.no_grad():
                    if self.manifold is None:
                        sims = torch.matmul(f, f.t()) # b*b
                    else:
                        # Use negative squared geodesic distances, mirroring poincare-embeddings.
                        sims = -self.manifold.dist(
                            f.unsqueeze(1),
                            f.unsqueeze(0),
                            dim=-1,
                        ).pow(2)
                    topk = torch.sort(sims, dim=1, descending=True).indices[:, :self.k] # b*k
                    nn = torch.zeros_like(sims).cuda()
                    rows = torch.arange(nn.size(0)).unsqueeze(1).expand_as(topk)
                    nn[rows, topk] = 1  
                    reciprocal = nn * nn.t() # b*b
                    reciprocal_w, reciprocal_s = reciprocal.chunk(2)

                    intersection = torch.matmul(reciprocal_s, reciprocal_w.t()) # b*b
                    union = reciprocal_s.sum(1,keepdim=True) + reciprocal_w.sum(1,keepdim=True).t() - intersection
                    weight = intersection / (union+1e-6) # b*b

                # loss
                loss_ce = self.criterion_ce(logits_w, pids) if self.use_ce else torch.tensor(0.0).cuda()

                if self.use_aug_ce:
                    loss_ce += self.criterion_ce(logits_s, pids) 
                
                alpha_value = self._get_alpha_value()

                loss_tri = self.criterion_tri(emb_w, pids, alpha=alpha_value) if self.use_triplet else torch.tensor(0.0).cuda()

                loss_align = self.align_loss(f_w_align, f_s_align, pids, weight, alpha_value) if self.use_align else torch.tensor(0.0).cuda()
                loss_drift_align = torch.tensor(0.0).cuda()

                if self.use_drift_align:
                    if f_w_drift is None or f_s_drift is None:
                        raise ValueError("Drift features not found for drift alignment loss.")
                    loss_drift_align = self.align_loss(f_w_drift, f_s_drift, pids, weight, alpha_value)

                total_align = loss_align + loss_drift_align
                
                loss_uniform = 0.5*(self.uniform_loss(f_w, alpha_value) + self.uniform_loss(f_s, alpha_value)) if self.use_uniform else torch.tensor(0.0).cuda()
                
                loss_domain = 0.5*(self.domain_loss(f_w, self.memory_bank.features, dids, alpha_value) +
                                   self.domain_loss(f_s, self.memory_bank.features, dids, alpha_value)) if self.use_domain else torch.tensor(0.0).cuda()

                if self.use_omega_reg and f_w_drift is not None and f_s_drift is not None:
                    loss_omega = self.omega_reg_weight * 0.5 * (self.omega_loss(f_w_drift) + self.omega_loss(f_s_drift))
                else:
                    loss_omega = torch.tensor(0.0, device=f.device)

                if domain_token_logits is not None and self.domain_token_loss_weight > 0.0:
                    loss_domain_token = self.domain_token_loss_weight * self.criterion_domain_token(domain_token_logits, domain_ids)
                else:
                    loss_domain_token = torch.tensor(0.0, device=f.device)

                with torch.no_grad():
                    self.memory_bank.momentum_update(f_w, pids)
                    
                loss = loss_ce + loss_tri + self.lam * total_align + loss_uniform + loss_domain + loss_omega + loss_domain_token

            if torch.isnan(loss):  # early exit if instability appears
                msg = f"NaN loss detected at epoch {epoch}, iter {i}"
                print(msg, flush=True)
                raise RuntimeError(msg)

            # update
            self.scaler.scale(loss).backward()
            # Unscale before clipping so grad norms are in true scale
            self.scaler.unscale_(optimizer)
            # Clip drift head gradients to prevent magnitude explosion
            if self.identity_dim is not None:
                module = getattr(self.model, 'module', self.model)
                drift_head = getattr(module, 'drift_head', None)
                if drift_head is not None:
                    torch.nn.utils.clip_grad_norm_(drift_head.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
            # loss.backward()
            # optimizer.step()
            
            # summing-up
            prec, = accuracy(logits.data, torch.cat([pids, pids]).data)
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_align.update(loss_align.item())
            losses_drift_align.update(loss_drift_align.item())
            losses_uniform.update(loss_uniform.item())
            losses_domain.update(loss_domain.item())
            losses_domain_token.update(loss_domain_token.item())
            losses_omega.update(loss_omega.item())
            precisions.update(prec[0])

            batch_time.update(time.time() - end)
            end = time.time()

            # Lightweight W&B logging at step granularity (if initialized)
            if wandb is not None and getattr(wandb, "run", None) is not None:
                try:
                    # use the first param group lr as representative
                    current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else None
                    global_step = epoch * iters + (i + 1)
                    wandb.log({
                        'train/loss_ce': losses_ce.val,
                        'train/loss_tri': losses_tri.val,
                        'train/loss_align': losses_align.val,
                        'train/loss_drift_align': losses_drift_align.val,
                        'train/loss_uniform': losses_uniform.val,
                        'train/loss_domain': losses_domain.val,
                        'train/loss_domain_token': losses_domain_token.val,
                        'train/loss_omega': losses_omega.val,
                        'train/loss_total': (losses_ce.val + losses_tri.val + self.lam * losses_align.val + losses_uniform.val + losses_domain.val + losses_domain_token.val + losses_omega.val),
                        'train/precision': precisions.val / 100.0 if precisions.val > 1 else precisions.val,  # ensure fraction
                        'time/batch_sec': batch_time.val,
                        'lr': current_lr,
                        'scaler/scale': float(self.scaler.get_scale()),
                        'epoch': epoch,
                        'iter': i + 1,
                        'omega_norm_mean': drift_norm_mean.item() if drift_norm_mean is not None else None,
                        'omega_norm_max': drift_norm_max.item() if drift_norm_max is not None else None,
                    }, step=global_step)
                except Exception:
                    pass  # never let logging break training

            if (i + 1) % print_freq == 0:
                print(f'Epoch: {epoch} [{i + 1}/{iters}] \t'
                      f' Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      f' L_CE: {losses_ce.val:.3f} ({losses_ce.avg:.3f})  '
                      f' L_Tri: {losses_tri.val:.3f} ({losses_tri.avg:.3f})  '
                      f' L_Align: {losses_align.val:.3f} ({losses_align.avg:.3f})  '
                      f' L_Uniform: {losses_uniform.val:.3f} ({losses_uniform.avg:.3f})  '
                      f' L_DTok: {losses_domain_token.val:.3f} ({losses_domain_token.avg:.3f})  '
                      f' L_Omega: {losses_omega.val:.3f} ({losses_omega.avg:.3f})  '
                      f' L_Domain: {losses_domain.val:.3f} ({losses_domain.avg:.3f})  '
                      f' Prec: {precisions.val:.2%} ({precisions.avg:.2%})')
                torch.cuda.empty_cache()

    def _parse_data(self, data):
        imgs_w, imgs_s, pids, dids, = data
        return imgs_w.cuda(), imgs_s.cuda(), pids.cuda(), dids.cuda()
    
    def align_loss(self, f_w, f_s, y, w, alpha=None):

        # For Finsler models, alignment loss operates ONLY on the identity part.
        # The drift vector models domain shift, not augmentation shift —
        # forcing alignment on drift creates conflicting gradients with
        # triplet/domain/uniform losses that rely on the asymmetric drift.
        # effective_dist_func = self.dist_func
        effective_dist_func = euclidean_dist
        alpha = None  # alpha is meaningless on identity-only features

        # Case 2: Finsler spaces and Euclidean Manifold
        if self.manifold is None:
            dist_squared = effective_dist_func(f_s, f_w, alpha=alpha).pow(2)

        # Case 3: Non-euclidean manifold
        elif self.manifold is not None:
            # Match the hyperbolic weighting strategy seen in geoopt/vision examples.
            dist_squared = self.manifold.dist(
                f_s.unsqueeze(1),
                f_w.unsqueeze(0),
                dim=-1,
            ).pow(2)
    
        else:
            raise ValueError("Invalid manifold/alpha configuration for align loss.")

        y = y.unsqueeze(-1) # b*1
        mask = torch.eq(y, y.T).to(f_s.device) # positive pairs
        align_loss = (dist_squared[mask] * w[mask]).sum() / w[mask].sum()
        return align_loss

    def uniform_loss(self, f, alpha=None):
        
        effective_dist_func = self.dist_func
        # For Finsler models with concatenated [identity|drift], use the full
        # embedding so the asymmetric drift contributes to the repulsion signal.
        if self.identity_dim is not None and effective_dist_func is finsler_drift_dist and f.size(1) <= self.identity_dim:
            effective_dist_func = euclidean_dist

        # Case 2: Finsler spaces and Euclidean Manifold
        if self.manifold is None:
            pairwise = effective_dist_func(f, f, alpha)  # (B, B)
            idx = torch.triu_indices(
                pairwise.size(0), 
                pairwise.size(1), 
                offset=1, 
                device=f.device,
            )
            pairwise = pairwise[idx[0], idx[1]]
            u_loss = pairwise.pow(2).mul(-2).exp().mean().log()

        # Case 3: Non-euclidean manifold
        elif self.manifold is not None:
            pairwise = self.manifold.dist(
                f.unsqueeze(1),
                f.unsqueeze(0),
                dim=-1,
            )
            idx = torch.triu_indices(
                pairwise.size(0),
                pairwise.size(1),
                offset=1,
                device=f.device,
            )
            pairwise = pairwise[idx[0], idx[1]]
            u_loss = pairwise.pow(2).mul(-2).exp().mean().log()

        else:
            raise ValueError("Invalid manifold/alpha configuration for uniform loss.")
        return u_loss

    def domain_loss(self, f, c, dids, alpha=None):
        m, n = f.size(0), c.size(0)
        
        effective_dist_func = self.dist_func
        # For Finsler models with concatenated [identity|drift], use the full
        # embedding so the asymmetric drift contributes to the domain repulsion.
        if self.identity_dim is not None and effective_dist_func is finsler_drift_dist and f.size(1) <= self.identity_dim:
            effective_dist_func = euclidean_dist

        # Case 2: Finsler spaces and Euclidean Manifold
        if self.manifold is None:
            dist_squared = effective_dist_func(f, c, alpha).pow(2)

        # Case 3: Non-euclidean manifold
        elif self.manifold is not None:
            chunk_size = self.manifold_chunk_size
            if chunk_size is None:
                dist_squared = self.manifold.dist(
                    f.unsqueeze(1),
                    c.unsqueeze(0),
                    dim=-1,
                ).pow(2)
            else:
                if chunk_size <= 0:
                    raise ValueError("manifold_chunk_size must be None or a positive integer")
                dist_squared = torch.empty(m, n, device=f.device, dtype=f.dtype)
                f_unsqueezed = f.unsqueeze(1)
                for i in range(0, n, chunk_size):
                    end = min(i + chunk_size, n)
                    c_chunk = c[i:end]
                    dist_squared_chunk = self.manifold.dist(
                        f_unsqueezed,
                        c_chunk.unsqueeze(0),
                        dim=-1,
                    ).pow(2)
                    dist_squared[:, i:end] = dist_squared_chunk

        else:
            raise ValueError("Invalid manifold/alpha configuration for domain loss.")

        domain_mask = torch.eq(dids.unsqueeze(-1), self.memory_bank.labels).float().cuda() # same domain
        sorted_dist, indices = torch.sort(dist_squared + (9999999.) * (1-domain_mask), dim=1, descending=False)

        # Explicitly delete the large distance matrix to free memory for the next steps
        del dist_squared
        torch.cuda.empty_cache()

        sorted_dist = sorted_dist[:, 1:m+1] # different class
        return sorted_dist.mul(-2).exp().mean().log()


    def omega_loss(self, omega):
        # Regularization on the omega drift vector to mitigate divergence to the maximum norm configured
           
        # If the finsler model was not selected -> skip omega regularization
        if omega is None or omega.numel() == 0:
            module = getattr(self.model, 'module', self.model)
            device = next(module.parameters()).device
            return torch.tensor(0.0, device=device)

        module = getattr(self.model, 'module', self.model)
        drift_head = getattr(module, 'drift_head', None)
        if drift_head is None or not hasattr(drift_head, 'max_norm'):
            return torch.tensor(0.0, device=omega.device)

        norms = torch.norm(omega, p=2, dim=1, keepdim=True)

        # Alternative 1: Log-Barrier Penalty
        max_norm = float(drift_head.max_norm)
        eps = 1e-6
        slack = torch.clamp(max_norm - norms, min=eps)
        loss = -torch.log(slack).mean()

        # Alternative 2: Margin-based Hinge Penalty
        # margin = 0.50
        # return F.relu(norms - margin).mean()

        # Alternative 3: Direct L2 Penalty
        # return norms.pow(2).mean()
        return loss