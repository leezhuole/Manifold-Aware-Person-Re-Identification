from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_monotonicity_pairs(pids, severities):
    """Number of ordered pairs (i, j) with same PID and severity[i] < severity[j]."""
    pids = pids.view(-1)
    severities = severities.view(-1)
    if pids.numel() < 2:
        return 0
    pid_eq = pids.unsqueeze(0) == pids.unsqueeze(1)
    sev_ordered = severities.unsqueeze(1) < severities.unsqueeze(0)
    return int((pid_eq & sev_ordered).sum().item())


class MonotonicityLoss(nn.Module):
    """Hinge loss enforcing higher scalar :math:`\\theta` for lower corruption severity.

    For each ordered pair ``(i, j)`` with the same PID and ``severity[i] < severity[j]``
    (cleaner index ``i``, more corrupted ``j``), we want ``theta[i] >= theta[j] + margin``.
    The per-pair penalty is ``[margin - (theta[i] - theta[j])]_+`` (same as
    ``[theta[j] - theta[i] + margin]_+``).

    Args:
        margin (float): minimum gap ``theta[low_sev] - theta[high_sev]`` before the hinge
            is inactive.
    """

    def __init__(self, margin=0.1):
        super(MonotonicityLoss, self).__init__()
        self.margin = float(margin)

    def forward(self, theta, pids, severities):
        """
        Args:
            theta: (B,) or (B, 1) tensor of per-example scalars.
            pids: (B,) integer person IDs (same device / dtype as typical CE targets).
            severities: (B,) integer severities in ``{0, ..., K-1}``.

        Returns:
            Scalar mean hinge over all valid ordered pairs; ``0`` if there are none
            (still on ``theta``'s device / dtype for autograd when ``theta`` requires grad).
        """
        theta = theta.view(-1)
        pids = pids.view(-1)
        severities = severities.view(-1)
        if theta.shape[0] != pids.shape[0] or theta.shape[0] != severities.shape[0]:
            raise ValueError(
                "theta, pids, severities must have the same length, got {}, {}, {}".format(
                    theta.shape[0], pids.shape[0], severities.shape[0]
                )
            )
        b = theta.shape[0]
        if b < 2:
            return theta.sum() * 0.0

        pid_eq = pids.unsqueeze(0) == pids.unsqueeze(1)
        # [i, j] True iff severity[i] < severity[j] (broadcast: (B,1) vs (1,B))
        sev_ordered = severities.unsqueeze(1) < severities.unsqueeze(0)
        mask = pid_eq & sev_ordered
        # [i, j] = theta[i] - theta[j]; penalize if below margin for sev[i] < sev[j]
        tdiff = theta.unsqueeze(1) - theta.unsqueeze(0)
        hinge = F.relu(self.margin - tdiff)
        valid = hinge[mask]
        if valid.numel() == 0:
            return theta.sum() * 0.0
        return valid.mean()
