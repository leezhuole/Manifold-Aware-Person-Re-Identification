# encoding: utf-8
"""Unit tests for PLAN.md Step 6: ``extract_features(..., return_theta)`` and
``bidirectional_evaluate`` (``bau/evaluators.py``).
"""

from __future__ import absolute_import, print_function

import os
import os.path as osp
import sys
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bau.evaluators import (  # noqa: E402
    bidirectional_evaluate,
    compute_mean_asymmetric_gap_d8,
    extract_features,
    pairwise_distance,
)
from bau.evaluation_metrics.ranking import mean_ap  # noqa: E402
from bau.utils.randers import randers_distance_matrix  # noqa: E402
from bau.utils.data.preprocessor import Preprocessor  # noqa: E402


class _EvalToyNet(nn.Module):
    """Deterministic (f_norm, theta) in eval; f from image mean, theta from first RGB channel sum."""

    def __init__(self, dim=16):
        super(_EvalToyNet, self).__init__()
        self.dim = dim
        self.proj = nn.Linear(3, dim, bias=False)
        nn.init.eye_(self.proj.weight[:, :3])
        self.theta_head = nn.Linear(dim, 1, bias=False)
        nn.init.constant_(self.theta_head.weight, 0.01)

    def forward(self, x):
        # x: (B, 3, H, W)
        b = x.size(0)
        pooled = x.view(b, 3, -1).mean(dim=2)
        emb = self.proj(pooled)
        f = F.normalize(emb, dim=1)
        theta = self.theta_head(emb)
        if self.training:
            return emb, f, theta
        return f, theta


def _write_rgb(path, rgb):
    os.makedirs(osp.dirname(path), exist_ok=True)
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[:, :] = rgb
    Image.fromarray(arr).save(path)


class TestExtractFeaturesTheta(unittest.TestCase):
    def setUp(self):
        self.root = osp.join(_ROOT, "tests", "_tmp_eval_theta")
        os.makedirs(self.root, exist_ok=True)
        self.paths = []
        for i, rgb in enumerate([(10, 0, 0), (20, 0, 0), (0, 30, 0)]):
            p = osp.join(self.root, "im{}.jpg".format(i))
            _write_rgb(p, rgb)
            self.paths.append(p)

    def tearDown(self):
        for p in getattr(self, "paths", []):
            if osp.isfile(p):
                os.remove(p)
        if osp.isdir(self.root):
            try:
                os.rmdir(self.root)
            except OSError:
                pass

    def test_return_theta_requires_tuple_forward(self):
        class BadNet(nn.Module):
            def forward(self, x):
                return x.mean()

        ds = [(self.paths[0], 0, 0)]
        loader = DataLoader(
            Preprocessor(ds, transform=lambda im: torch.zeros(3, 4, 4)),
            batch_size=1,
            shuffle=False,
        )
        with mock.patch("bau.evaluators.time.sleep", lambda *_: None):
            with self.assertRaises(TypeError):
                extract_features(BadNet(), loader, print_freq=10**9, return_theta=True)

    def test_return_theta_populates_dict(self):
        net = _EvalToyNet(dim=8)
        net.eval()
        ds = [(p, i, i % 2) for i, p in enumerate(self.paths)]
        loader = DataLoader(
            Preprocessor(ds, transform=lambda im: torch.zeros(3, 4, 4)),
            batch_size=2,
            shuffle=False,
        )
        with mock.patch("bau.evaluators.time.sleep", lambda *_: None):
            feats, labs, thetas = extract_features(
                net, loader, print_freq=10**9, return_theta=True
            )
        self.assertEqual(len(feats), 3)
        self.assertEqual(len(thetas), 3)
        for p in self.paths:
            self.assertIn(p, thetas)
            self.assertIsInstance(thetas[p], float)


class TestBidirectionalEvaluate(unittest.TestCase):
    def setUp(self):
        self.root = osp.join(_ROOT, "tests", "_tmp_eval_bidir")
        os.makedirs(self.root, exist_ok=True)
        # Two identities, two "domains": clean vs corrupted filenames
        self.clean_a = osp.join(self.root, "clean_a.jpg")
        self.clean_b = osp.join(self.root, "clean_b.jpg")
        self.corr_a = osp.join(self.root, "corr_a.jpg")
        self.corr_b = osp.join(self.root, "corr_b.jpg")
        _write_rgb(self.clean_a, (40, 40, 40))
        _write_rgb(self.clean_b, (80, 80, 80))
        _write_rgb(self.corr_a, (41, 41, 41))
        _write_rgb(self.corr_b, (81, 81, 81))

    def tearDown(self):
        for p in [
            self.clean_a,
            self.clean_b,
            self.corr_a,
            self.corr_b,
        ]:
            if osp.isfile(p):
                os.remove(p)
        if osp.isdir(self.root):
            try:
                os.rmdir(self.root)
            except OSError:
                pass

    def _make_loader(self, rows):
        loader = DataLoader(
            Preprocessor(rows, transform=lambda im: torch.zeros(3, 4, 4)),
            batch_size=2,
            shuffle=False,
        )
        return loader

    def test_randers_alpha_zero_tensor_matches_pairwise(self):
        """α=0 Randers matrix equals ``pairwise_distance`` (θ term vanishes)."""
        net = _EvalToyNet(dim=8)
        net.eval()
        query_clean = [(self.clean_a, 0, 1), (self.clean_b, 1, 1)]
        gallery_corrupted = [(self.corr_a, 0, 2), (self.corr_b, 1, 2)]
        rows = query_clean + gallery_corrupted
        loader = self._make_loader(rows)

        with mock.patch("bau.evaluators.time.sleep", lambda *_: None):
            feats, _, thetas = extract_features(
                net, loader, print_freq=10**9, return_theta=True
            )
        f_q = torch.cat([feats[f].view(1, -1) for f, _, _ in query_clean], 0)
        f_g = torch.cat([feats[f].view(1, -1) for f, _, _ in gallery_corrupted], 0)
        t_q = torch.tensor([thetas[f] for f, _, _ in query_clean], dtype=f_q.dtype)
        t_g = torch.tensor(
            [thetas[f] for f, _, _ in gallery_corrupted], dtype=f_g.dtype
        )
        r0 = randers_distance_matrix(f_q, t_q, f_g, t_g, 0.0)
        d_sq, _, _ = pairwise_distance(feats, query_clean, gallery_corrupted)
        self.assertTrue(torch.allclose(r0, d_sq, atol=1e-4, rtol=1e-4))
        qids = [pid for _, pid, _ in query_clean]
        gids = [pid for _, pid, _ in gallery_corrupted]
        qc = [cam for _, _, cam in query_clean]
        gc = [cam for _, _, cam in gallery_corrupted]
        m_r = mean_ap(r0, qids, gids, qc, gc)
        m_e = mean_ap(d_sq, qids, gids, qc, gc)
        self.assertAlmostEqual(m_r, m_e, places=5)

    def test_bidirectional_evaluate_structure_and_flat_theta(self):
        net = _EvalToyNet(dim=8)
        net.eval()
        query_clean = [(self.clean_a, 0, 1), (self.clean_b, 1, 1)]
        gallery_corrupted = [(self.corr_a, 0, 2), (self.corr_b, 1, 2)]
        query_corrupted = list(gallery_corrupted)
        gallery_clean = list(query_clean)
        rows = query_clean + gallery_corrupted
        loader = self._make_loader(rows)

        with mock.patch("bau.evaluators.time.sleep", lambda *_: None):
            out = bidirectional_evaluate(
                net,
                loader,
                query_clean,
                gallery_corrupted,
                query_corrupted,
                gallery_clean,
                alpha_values=[0.0, 0.3],
                return_theta=True,
                print_freq=10**9,
                verbose=False,
            )
        self.assertIn("euclidean", out)
        self.assertIn("randers", out)
        self.assertIn("delta_mAP", out["euclidean"])
        self.assertEqual(len(out["randers"]), 2)
        for a in (0.0, 0.3):
            self.assertIn(a, out["randers"])
            block = out["randers"][a]
            self.assertIn("direction_A", block)
            self.assertIn("direction_B", block)
            self.assertIn("delta_mAP", block)
            self.assertIn("d8_mean_asymmetric_gap", block)

    def test_zero_theta_randers_invariant_in_alpha(self):
        """With return_theta=False, θ treated as 0 → Randers term vanishes; mAP same for all α."""
        net = _EvalToyNet(dim=8)
        net.eval()
        query_clean = [(self.clean_a, 0, 1), (self.clean_b, 1, 1)]
        gallery_corrupted = [(self.corr_a, 0, 2), (self.corr_b, 1, 2)]
        query_corrupted = list(gallery_corrupted)
        gallery_clean = list(query_clean)
        rows = query_clean + gallery_corrupted
        loader = self._make_loader(rows)

        with mock.patch("bau.evaluators.time.sleep", lambda *_: None):
            out = bidirectional_evaluate(
                net,
                loader,
                query_clean,
                gallery_corrupted,
                query_corrupted,
                gallery_clean,
                alpha_values=[0.0, 0.5, 0.9],
                return_theta=False,
                print_freq=10**9,
                verbose=False,
            )
        m0 = out["randers"][0.0]["direction_A"]["mAP"]
        m5 = out["randers"][0.5]["direction_A"]["mAP"]
        m9 = out["randers"][0.9]["direction_A"]["mAP"]
        self.assertAlmostEqual(m0, m5, places=5)
        self.assertAlmostEqual(m0, m9, places=5)


class TestD8MeanAsymmetricGap(unittest.TestCase):
    """Mean asymmetric Randers gap on PID + severity-matched pairs (camera may differ)."""

    def test_analytic_gap_identical_features(self):
        """Same ``f`` for z^0 and z^k ⇒ Euclidean term cancels in the gap; only θ remains."""
        root = osp.join(_ROOT, "tests", "_tmp_d8")
        os.makedirs(root, exist_ok=True)
        try:
            p0 = osp.join(root, "0001_c1s1_000001_01.jpg")
            pk = osp.join(root, "0001_c1s2_000001_01.jpg")
            _write_rgb(p0, (1, 2, 3))
            _write_rgb(pk, (4, 5, 6))
            f = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
            features = {p0: f.clone(), pk: f.clone()}
            theta_dict = {p0: 0.2, pk: 0.5}
            alpha = 0.25
            out = compute_mean_asymmetric_gap_d8(
                features,
                theta_dict,
                gallery_clean=[(p0, 0, 1)],
                gallery_corrupted=[(pk, 0, 1)],
                alpha=alpha,
                severity_levels=(1,),
            )
            expected = 2.0 * alpha * (0.5 - 0.2)
            self.assertEqual(out["n_pairs"], 1)
            self.assertAlmostEqual(out["mean_gap"], expected, places=6)
            self.assertAlmostEqual(out["mean_gap_by_severity"][1], expected, places=6)
        finally:
            for p in [osp.join(root, "0001_c1s1_000001_01.jpg"), osp.join(root, "0001_c1s2_000001_01.jpg")]:
                if osp.isfile(p):
                    os.remove(p)
            if osp.isdir(root):
                os.rmdir(root)

    def test_theta_none_gives_zero_gap(self):
        root = osp.join(_ROOT, "tests", "_tmp_d8b")
        os.makedirs(root, exist_ok=True)
        try:
            p0 = osp.join(root, "0002_c2s1_000001_01.jpg")
            pk = osp.join(root, "0002_c2s3_000001_01.jpg")
            _write_rgb(p0, (1, 0, 0))
            _write_rgb(pk, (0, 1, 0))
            features = {
                p0: torch.tensor([[1.0, 0.0]]),
                pk: torch.tensor([[0.0, 1.0]]),
            }
            out = compute_mean_asymmetric_gap_d8(
                features,
                None,
                gallery_clean=[(p0, 5, 2)],
                gallery_corrupted=[(pk, 5, 2)],
                alpha=0.9,
                severity_levels=(2,),
            )
            self.assertEqual(out["n_pairs"], 1)
            self.assertAlmostEqual(out["mean_gap"], 0.0, places=6)
        finally:
            for p in [osp.join(root, "0002_c2s1_000001_01.jpg"), osp.join(root, "0002_c2s3_000001_01.jpg")]:
                if osp.isfile(p):
                    os.remove(p)
            if osp.isdir(root):
                os.rmdir(root)

    def test_cross_camera_same_pid_pairs(self):
        """Different cam_id, same PID and severities 0 vs k — must pair after relaxing same-cam."""
        root = osp.join(_ROOT, "tests", "_tmp_d8_crosscam")
        os.makedirs(root, exist_ok=True)
        try:
            p0 = osp.join(root, "0042_c1s1_000001_01.jpg")
            pk = osp.join(root, "0042_c2s2_000001_01.jpg")
            _write_rgb(p0, (1, 2, 3))
            _write_rgb(pk, (4, 5, 6))
            f = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
            features = {p0: f.clone(), pk: f.clone()}
            theta_dict = {p0: 0.1, pk: 0.4}
            alpha = 0.5
            out = compute_mean_asymmetric_gap_d8(
                features,
                theta_dict,
                gallery_clean=[(p0, 42, 1)],
                gallery_corrupted=[(pk, 42, 2)],
                alpha=alpha,
                severity_levels=(1,),
            )
            expected = 2.0 * alpha * (0.4 - 0.1)
            self.assertEqual(out["n_pairs"], 1)
            self.assertAlmostEqual(out["mean_gap"], expected, places=6)
        finally:
            for p in [osp.join(root, "0042_c1s1_000001_01.jpg"), osp.join(root, "0042_c2s2_000001_01.jpg")]:
                if osp.isfile(p):
                    os.remove(p)
            if osp.isdir(root):
                os.rmdir(root)


if __name__ == "__main__":
    unittest.main()
