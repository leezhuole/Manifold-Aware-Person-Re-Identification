# encoding: utf-8
"""Unit tests for ``bau.utils.randers`` (PLAN.md Step 5)."""

from __future__ import absolute_import, print_function

import unittest

import torch

from bau.utils.randers import randers_distance_matrix


def _pairwise_sq_reference(f_q, f_g):
    """Mirror ``bau.evaluators.pairwise_distance`` tensor math (squared L2)."""
    m, n = f_q.size(0), f_g.size(0)
    x = f_q.view(m, -1)
    y = f_g.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(x, y.t(), beta=1.0, alpha=-2.0)
    return torch.clamp(dist_m, min=0.0)


class TestRandersDistanceMatrix(unittest.TestCase):
    def test_alpha_zero_matches_pairwise_sq(self):
        """α=0 and arbitrary θ: Euclidean term equals squared L2 from evaluators."""
        torch.manual_seed(0)
        f_q = torch.randn(4, 16)
        f_g = torch.randn(7, 16)
        theta_q = torch.randn(4)
        theta_g = torch.randn(7)
        sq = _pairwise_sq_reference(f_q, f_g)
        d0 = randers_distance_matrix(f_q, theta_q, f_g, theta_g, 0.0)
        self.assertTrue(torch.allclose(d0, sq, atol=1e-5, rtol=1e-5))

    def test_theta_term_broadcast_manual(self):
        """dist[i,j] = sq[i,j] + alpha * (theta_g[j] - theta_q[i])."""
        f_q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        f_g = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        theta_q = torch.tensor([0.0, 1.0])
        theta_g = torch.tensor([2.0, 3.0])
        alpha = 0.5
        d = randers_distance_matrix(f_q, theta_q, f_g, theta_g, alpha)
        sq = _pairwise_sq_reference(f_q, f_g)
        manual = sq + alpha * (theta_g.unsqueeze(0) - theta_q.unsqueeze(1))
        self.assertTrue(torch.allclose(d, manual))

    def test_asymmetric_swap_query_gallery(self):
        """Swapping (f_q,theta_q) with (f_g,theta_g) changes the matrix when alpha > 0."""
        torch.manual_seed(1)
        f_q = torch.randn(3, 32)
        f_g = torch.randn(5, 32)
        theta_q = torch.randn(3)
        theta_g = torch.randn(5)
        alpha = 0.3
        d_ab = randers_distance_matrix(f_q, theta_q, f_g, theta_g, alpha)
        d_ba = randers_distance_matrix(f_g, theta_g, f_q, theta_q, alpha)
        self.assertFalse(torch.allclose(d_ab, d_ba.t()))
        self.assertGreater((d_ab - d_ba.t()).abs().max().item(), 1e-6)

    def test_device_dtype_preserved(self):
        """Output lives on the same device as f_q (CPU test)."""
        f_q = torch.randn(2, 4, dtype=torch.float64)
        f_g = torch.randn(3, 4, dtype=torch.float64)
        theta_q = torch.randn(2, dtype=torch.float64)
        theta_g = torch.randn(3, dtype=torch.float64)
        d = randers_distance_matrix(f_q, theta_q, f_g, theta_g, 0.1)
        self.assertEqual(d.dtype, torch.float64)
        self.assertEqual(d.device, f_q.device)

    def test_alpha_scalar_tensor(self):
        alpha = torch.tensor(0.25)
        f_q = torch.ones(1, 2)
        f_g = torch.zeros(1, 2)
        theta_q = torch.tensor([1.0])
        theta_g = torch.tensor([0.0])
        d = randers_distance_matrix(f_q, theta_q, f_g, theta_g, alpha)
        # Squared L2: ||[0,0]-[1,1]||^2 = 2; theta term = 0.25 * (0 - 1) = -0.25
        self.assertAlmostEqual(d[0, 0].item(), 1.75, places=5)

    def test_theta_length_mismatch_raises(self):
        f_q = torch.randn(2, 3)
        f_g = torch.randn(2, 3)
        with self.assertRaises(ValueError):
            randers_distance_matrix(f_q, torch.zeros(3), f_g, torch.zeros(2), 0.1)


if __name__ == "__main__":
    unittest.main()
