# encoding: utf-8
"""Unit tests for ``bau.models.model.resnet50`` optional θ head (PLAN.md Step 3)."""

from __future__ import absolute_import, print_function

import os.path as osp
import sys
import unittest

import torch

_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bau.models.model import resnet50  # noqa: E402


class TestResnet50ThetaHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _tiny_input(self, batch=2):
        return torch.randn(batch, 3, 224, 224, device=self.device)

    def test_default_matches_legacy_contract(self):
        """Training returns (emb, f_norm, logits); eval returns f_norm only."""
        m = resnet50(num_classes=10, pretrained=False, with_theta_head=False).to(self.device)
        m.train()
        x = self._tiny_input()
        emb, f_norm, logits = m(x)
        self.assertEqual(emb.shape, (2, 2048))
        self.assertEqual(f_norm.shape, (2, 2048))
        self.assertEqual(logits.shape, (2, 10))
        self.assertTrue(torch.allclose(f_norm.norm(dim=1), torch.ones(2, device=self.device), atol=1e-5))
        m.eval()
        out = m(x)
        self.assertEqual(out.shape, (2, 2048))

    def test_theta_head_not_created_when_disabled(self):
        m = resnet50(num_classes=0, pretrained=False, with_theta_head=False)
        self.assertFalse(hasattr(m, "theta_head"))

    def test_theta_head_train_eval_shapes(self):
        m = resnet50(num_classes=0, pretrained=False, with_theta_head=True).to(self.device)
        x = self._tiny_input(batch=3)
        m.train()
        emb, f_norm, theta = m(x)
        self.assertEqual(emb.shape, (3, 2048))
        self.assertEqual(f_norm.shape, (3, 2048))
        self.assertEqual(theta.shape, (3, 1))
        m.eval()
        f_e, th_e = m(x)
        self.assertEqual(f_e.shape, (3, 2048))
        self.assertEqual(th_e.shape, (3, 1))
        z = torch.cat([f_e, th_e], dim=1)
        self.assertEqual(z.shape, (3, 2049))

    def test_theta_linear_in_emb_not_in_f_norm(self):
        """Loss depending only on θ must not backprop into ``bn_neck`` parameters."""
        m = resnet50(num_classes=0, pretrained=False, with_theta_head=True).to(self.device)
        m.train()
        x = self._tiny_input(batch=2)
        _, _, theta = m(x)
        loss = theta.sum()
        loss.backward()
        for p in m.bn_neck.parameters():
            self.assertTrue(
                p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad)),
                "bn_neck should not receive gradient from theta-only loss",
            )
        self.assertIsNotNone(m.theta_head.weight.grad)

    def test_freeze_pretrained_leaves_theta_trainable(self):
        m = resnet50(num_classes=100, pretrained=False, with_theta_head=True).to(self.device)
        m.freeze_pretrained()
        for name, p in m.named_parameters():
            if "theta_head" in name:
                self.assertTrue(p.requires_grad, name)
            elif "theta_head" not in name:
                self.assertFalse(p.requires_grad, name)

    def test_theta_head_init_small(self):
        m = resnet50(num_classes=0, pretrained=False, with_theta_head=True)
        w = m.theta_head.weight.data
        self.assertLess(w.abs().max().item(), 1e-2)

    def test_identical_f_norm_without_theta_vs_zero_theta_head(self):
        """θ head is a separate branch: same weights and zero ``theta_head`` ⇒ same ``f_norm``."""
        torch.manual_seed(0)
        m0 = resnet50(num_classes=0, pretrained=False, with_theta_head=False).to(self.device)
        torch.manual_seed(0)
        m1 = resnet50(num_classes=0, pretrained=False, with_theta_head=True).to(self.device)
        with torch.no_grad():
            m1.theta_head.weight.zero_()
        x = self._tiny_input()
        m0.eval()
        m1.eval()
        f0 = m0(x)
        f1, th1 = m1(x)
        self.assertTrue(torch.allclose(f0, f1, atol=1e-6))
        self.assertTrue(torch.allclose(th1, torch.zeros_like(th1)))


if __name__ == "__main__":
    unittest.main()
