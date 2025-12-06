from __future__ import absolute_import

from typing import cast

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torch


class MemoryBank(nn.Module):
    """
    Memory Bank module for storing and updating feature representations of samples.
    Supports optional manifold-aware updates (e.g., for Poincare ball embeddings).

    Note:
    (1) Because the momentum updates are additive, we can only compute the addition in 
    a linear Euclidean space.
    
    """
    def __init__(self, num_features, num_samples, momentum=0.1, manifold=None, eps=1e-12):
        """
        Docstring for __init__
        
        :param num_features: Embedding dimensionality of selected model
        :param num_samples: Number of samples to store in the memory bank
        :param momentum: Momentum factor for updating features
        :param manifold: Manifold object for manifold-aware updates (optional)
        :param eps: Small epsilon value to avoid division by zero
        """
        super(MemoryBank, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.manifold = manifold
        self.eps = eps

        self.register_buffer('features', torch.zeros(num_samples, num_features, dtype=torch.float))
        self.register_buffer('labels', torch.zeros(num_samples, num_features, dtype=torch.long))

    def momentum_update(self, inputs, targets):
        features = cast(torch.Tensor, self.features)
        for x, y in zip(inputs, targets):
            if self.manifold is None:
                updated = self.momentum * features[y] + (1. - self.momentum) * x
                updated_norm = updated.norm().clamp_min(self.eps)
                features[y] = updated / updated_norm
            else:
                # Update in the tangent space at the origin, mirroring poincare-embeddings momentum.
                current = features[y].unsqueeze(0)
                current = self.manifold.projx(current, dim=-1)                         # Project onto the ball
                current_tan = self.manifold.logmap0(current, dim=-1)                   # Log map to tangent space

                new_tan = self.manifold.logmap0(x.unsqueeze(0), dim=-1)                # Log map to tangent space
                mixed = self.momentum * current_tan + (1. - self.momentum) * new_tan   # Apply exponential moving average

                updated = self.manifold.expmap0(mixed, dim=-1)                         # Map back
                updated = self.manifold.projx(updated, dim=-1)                         # Project onto the ball
                features[y] = updated.squeeze(0)
            self.features[y] = features[y]