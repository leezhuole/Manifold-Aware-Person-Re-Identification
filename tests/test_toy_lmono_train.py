# encoding: utf-8
"""Unit tests for PLAN.md Step 7 (train script helpers, evaluators, preprocessor)."""

from __future__ import absolute_import

import os
import tempfile
import unittest

import torch
from torch import nn
from PIL import Image

from bau.evaluators import (
    _resolve_feature_key,
    bidirectional_evaluate,
    toy_severity_from_reid_fname,
)
from bau.loss import count_monotonicity_pairs
from bau.models import create as create_model
from bau.utils.data.preprocessor import SeverityPreprocessor


class TestToySeverityFromFname(unittest.TestCase):
    def test_parse(self):
        p = "/data/ToyCorruption/gallery/0001_c1s3_000001_01.jpg"
        self.assertEqual(toy_severity_from_reid_fname(p), 2)


class TestResolveFeatureKey(unittest.TestCase):
    def test_basename_fallback(self):
        d = {"/a/b/0001_c1s2_000001_01.jpg": 1.0}
        self.assertEqual(
            _resolve_feature_key(d, "/other/root/0001_c1s2_000001_01.jpg"),
            "/a/b/0001_c1s2_000001_01.jpg",
        )


class TestCountMonotonicityPairs(unittest.TestCase):
    def test_zero_when_single_severity(self):
        pids = torch.tensor([0, 0, 0])
        sev = torch.tensor([1, 1, 1])
        self.assertEqual(count_monotonicity_pairs(pids, sev), 0)

    def test_ordered_pairs(self):
        pids = torch.tensor([0, 0])
        sev = torch.tensor([0, 1])
        self.assertEqual(count_monotonicity_pairs(pids, sev), 1)


class TestToyResnet50Forward(unittest.TestCase):
    def test_train_tuple_and_theta_from_emb(self):
        m = create_model("toy_resnet50", num_classes=0, pretrained=False)
        m.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            f, th = m(x)
        self.assertEqual(f.shape, (2, 2048))
        self.assertEqual(th.shape, (2, 1))
        m.train()
        emb, fn, th2 = m(x)
        self.assertEqual(emb.shape, (2, 2048))
        self.assertEqual(fn.shape, (2, 2048))
        self.assertEqual(th2.shape, (2, 1))

    def test_freeze_leaves_theta_trainable(self):
        m = create_model("toy_resnet50", num_classes=0, pretrained=False)
        m.freeze_pretrained()
        self.assertTrue(m.theta_head.weight.requires_grad)
        self.assertFalse(next(m.base.parameters()).requires_grad)


class TestSeverityPreprocessor(unittest.TestCase):
    def test_returns_fifth_severity(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = os.path.join(tmp, "im.jpg")
            Image.new("RGB", (32, 24), color=(120, 5, 200)).save(img_path)
            ds = [(img_path, 3, 1, 2)]
            sp = SeverityPreprocessor(ds, root=None, transform=None)
            img, fname, pid, cid, sev = sp[0]
            self.assertEqual(pid, 3)
            self.assertEqual(cid, 1)
            self.assertEqual(sev, 2)
            self.assertIsInstance(img, Image.Image)


class TestBidirectionalEvaluateTrivial(unittest.TestCase):
    """Tiny synthetic ranking: θ=0 so Randers α>0 matches Euclidean ranking."""

    def test_randers_alpha_matches_euclidean_when_theta_flat(self):
        class _M(nn.Module):
            def forward(self, x):
                b = x.size(0)
                f = torch.nn.functional.normalize(torch.randn(b, 4, device=x.device), dim=1)
                return f, torch.zeros(b, 1, device=x.device)

        query_clean = [("q0", 0, 0), ("q1", 1, 1)]
        gallery_c = [("g0", 0, 2), ("g1", 1, 3), ("g2", 2, 4)]
        union = query_clean + gallery_c

        class _DS(torch.utils.data.Dataset):
            def __init__(self, rows):
                self.rows = rows

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, i):
                _, pid, cam = self.rows[i]
                t = torch.zeros(3, 8, 8)
                return t, self.rows[i][0], pid, cam

        loader = torch.utils.data.DataLoader(
            _DS(union), batch_size=4, shuffle=False
        )
        model = _M()
        out = bidirectional_evaluate(
            model,
            loader,
            query_clean=query_clean,
            gallery_corrupted=gallery_c,
            query_corrupted=gallery_c,
            gallery_clean=query_clean,
            alpha_values=[0.0, 0.5],
            return_theta=True,
            print_freq=1000,
            verbose=False,
        )
        self.assertIn("theta_by_fname", out)
        eu = out["euclidean"]["direction_A"]["mAP"]
        r0 = out["randers"][0.0]["direction_A"]["mAP"]
        r05 = out["randers"][0.5]["direction_A"]["mAP"]
        self.assertAlmostEqual(eu, r0, places=5)
        self.assertAlmostEqual(r0, r05, places=5)


if __name__ == "__main__":
    unittest.main()
