# encoding: utf-8
"""Unit tests for ``MonotonicityLoss`` (PLAN.md Step 4)."""

from __future__ import absolute_import, print_function

import unittest

import torch

from bau.loss.mono import MonotonicityLoss


class TestMonotonicityLoss(unittest.TestCase):
    def test_satisfied_ordering_zero_loss(self):
        """Lower severity has sufficiently higher theta -> hinge inactive."""
        loss_fn = MonotonicityLoss(margin=0.1)
        # PID 0: sev 0 -> theta 1.0, sev 1 -> theta 0.0 (gap 1.0 > margin)
        theta = torch.tensor([1.0, 0.0], requires_grad=True)
        pids = torch.tensor([0, 0], dtype=torch.long)
        severities = torch.tensor([0, 1], dtype=torch.long)
        out = loss_fn(theta, pids, severities)
        self.assertAlmostEqual(out.item(), 0.0, places=6)

    def test_reversed_ordering_positive_loss(self):
        """Lower severity has lower theta -> positive hinge."""
        loss_fn = MonotonicityLoss(margin=0.1)
        theta = torch.tensor([0.0, 1.0], requires_grad=True)
        pids = torch.tensor([0, 0], dtype=torch.long)
        severities = torch.tensor([0, 1], dtype=torch.long)
        out = loss_fn(theta, pids, severities)
        # margin - (0 - 1) = 1.1
        self.assertAlmostEqual(out.item(), 1.1, places=5)

    def test_exact_margin_zero(self):
        """theta[low] - theta[high] == margin -> zero."""
        loss_fn = MonotonicityLoss(margin=0.2)
        theta = torch.tensor([0.5, 0.3], requires_grad=True)  # diff 0.2
        pids = torch.tensor([0, 0], dtype=torch.long)
        severities = torch.tensor([0, 1], dtype=torch.long)
        out = loss_fn(theta, pids, severities)
        self.assertAlmostEqual(out.item(), 0.0, places=6)

    def test_no_valid_pairs_returns_zero(self):
        """All different PIDs -> no (i,j) with same pid and ordered severities."""
        loss_fn = MonotonicityLoss(margin=0.1)
        theta = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
        pids = torch.tensor([0, 1, 2], dtype=torch.long)
        severities = torch.tensor([0, 1, 2], dtype=torch.long)
        out = loss_fn(theta, pids, severities)
        self.assertAlmostEqual(out.item(), 0.0, places=6)

    def test_multi_severity_pairs_mean(self):
        """Three severities same PID: pairs (0,1), (0,2), (1,2) with unit gaps."""
        loss_fn = MonotonicityLoss(margin=0.0)
        # theta decreasing with severity -> all hinges 0
        theta = torch.tensor([2.0, 1.0, 0.0], requires_grad=True)
        pids = torch.zeros(3, dtype=torch.long)
        severities = torch.tensor([0, 1, 2], dtype=torch.long)
        out = loss_fn(theta, pids, severities)
        self.assertAlmostEqual(out.item(), 0.0, places=6)

    def test_gradient_nonzero_when_violation(self):
        loss_fn = MonotonicityLoss(margin=0.1)
        theta = torch.tensor([0.0, 1.0], requires_grad=True)
        pids = torch.tensor([0, 0], dtype=torch.long)
        severities = torch.tensor([0, 1], dtype=torch.long)
        out = loss_fn(theta, pids, severities)
        out.backward()
        self.assertIsNotNone(theta.grad)
        self.assertGreater(theta.grad.abs().sum().item(), 0.0)

    def test_theta_b1_shape(self):
        loss_fn = MonotonicityLoss(margin=0.1)
        theta = torch.tensor([[1.0], [0.0]], requires_grad=True)
        pids = torch.tensor([0, 0], dtype=torch.long)
        severities = torch.tensor([0, 1], dtype=torch.long)
        out = loss_fn(theta, pids, severities)
        self.assertAlmostEqual(out.item(), 0.0, places=6)

    def test_length_mismatch_raises(self):
        loss_fn = MonotonicityLoss()
        theta = torch.zeros(2)
        pids = torch.zeros(3, dtype=torch.long)
        severities = torch.zeros(2, dtype=torch.long)
        with self.assertRaises(ValueError):
            loss_fn(theta, pids, severities)


if __name__ == "__main__":
    unittest.main()
