# encoding: utf-8
"""Randers-style asymmetric distance on L2 identity features plus scalar θ."""

from __future__ import absolute_import

import torch


def randers_distance_matrix(f_q, theta_q, f_g, theta_g, alpha):
    """Asymmetric query→gallery distance matrix.

    The identity term is **squared** L2, matching ``pairwise_distance`` in
    ``bau/evaluators.py`` (no square root). Full score:

    ``dist[i, j] = ||f_g[j]-f_q[i]||_2^2 + alpha * (theta_g[j] - theta_q[i])``.

    Parameters
    ----------
    f_q : Tensor
        ``(Nq, D)`` L2-normalized identity embeddings (queries).
    theta_q : Tensor
        ``(Nq,)`` scalar θ per query.
    f_g : Tensor
        ``(Ng, D)`` L2-normalized identity embeddings (gallery).
    theta_g : Tensor
        ``(Ng,)`` scalar θ per gallery row.
    alpha : float or Tensor
        Randers weight (scalar).

    Returns
    -------
    Tensor
        ``(Nq, Ng)`` distance matrix on the same device / dtype as ``f_q``.
    """
    if f_q.dim() != 2 or f_g.dim() != 2:
        raise ValueError("f_q and f_g must be 2-D tensors")
    m, d_q = f_q.size()
    n, d_g = f_g.size()
    if d_q != d_g:
        raise ValueError("Embedding dims must match, got {} vs {}".format(d_q, d_g))
    theta_q = theta_q.view(-1)
    theta_g = theta_g.view(-1)
    if theta_q.shape[0] != m or theta_g.shape[0] != n:
        raise ValueError(
            "theta_q length must match Nq ({}), theta_g length Ng ({})".format(
                theta_q.shape[0], theta_g.shape[0]
            )
        )

    # Euclidean term (same algebra as ``pairwise_distance`` in evaluators)
    dist_e = torch.pow(f_q, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(f_g, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_e.addmm_(1, -2, f_q, f_g.t())
    dist_e = torch.clamp(dist_e, min=0.0)

    a = alpha if torch.is_tensor(alpha) else torch.tensor(
        alpha, device=f_q.device, dtype=f_q.dtype
    )
    dist = dist_e + a * (theta_g.view(1, n) - theta_q.view(m, 1))
    return dist
