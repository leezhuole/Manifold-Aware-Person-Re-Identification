# encoding: utf-8
"""Unit tests for ``bau.datasets.toy_corruption.ToyCorruption`` (PLAN.md Step 1)."""

from __future__ import absolute_import, print_function

import json
import os
import os.path as osp
import sys
import unittest

# Repo root on path for `import bau`
_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bau.datasets import create  # noqa: E402
from bau.datasets.toy_corruption import ToyCorruption  # noqa: E402


def _write_minimal_dataset(root):
    """Tiny v4-shaped metadata + empty image files for path resolution checks."""
    meta = {
        "dataset": {
            "toy_dataset_version": "test",
            "train": {"num_identities": 2},
            "eval": {"num_identities": 1},
        },
        "images": {
            "train/0001_c1s1_000001_01.jpg": {
                "split": "train",
                "new_pid": 1,
                "source_idx": 1,
                "severity": 0,
                "cam_id": 1,
            },
            "train/0001_c1s2_000001_01.jpg": {
                "split": "train",
                "new_pid": 1,
                "source_idx": 1,
                "severity": 1,
                "cam_id": 1,
            },
            "train/0002_c1s1_000001_01.jpg": {
                "split": "train",
                "new_pid": 2,
                "source_idx": 1,
                "severity": 4,
                "cam_id": 1,
            },
            "eval/0001_c1s1_000001_01.jpg": {
                "split": "eval",
                "new_pid": 1,
                "source_idx": 1,
                "severity": 0,
                "cam_id": 1,
            },
            "eval/0001_c2s1_000001_01.jpg": {
                "split": "eval",
                "new_pid": 1,
                "source_idx": 2,
                "severity": 0,
                "cam_id": 2,
            },
            "eval/0001_c1s2_000001_01.jpg": {
                "split": "eval",
                "new_pid": 1,
                "source_idx": 1,
                "severity": 1,
                "cam_id": 1,
            },
        },
    }
    os.makedirs(root, exist_ok=True)
    bbox_train = osp.join(root, "bounding_box_train")
    q1 = osp.join(root, "query_s1")
    q2 = osp.join(root, "query_s2")
    gal = osp.join(root, "gallery")
    for d in (bbox_train, q1, q2, gal):
        os.makedirs(d, exist_ok=True)

    with open(osp.join(root, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    for rel in (
        "bounding_box_train/0001_c1s1_000001_01.jpg",
        "bounding_box_train/0001_c1s2_000001_01.jpg",
        "bounding_box_train/0002_c1s1_000001_01.jpg",
        "query_s1/0001_c1s1_000001_01.jpg",
        "query_s2/0001_c2s1_000001_01.jpg",
        "gallery/0001_c1s2_000001_01.jpg",
    ):
        open(osp.join(root, rel), "wb").close()


class TestToyCorruptionMinimal(unittest.TestCase):
    def setUp(self):
        self._root = osp.join(_ROOT, "tests", "_tmp_toy_minimal")
        if osp.isdir(self._root):
            for walk_root, dirs, files in os.walk(self._root, topdown=False):
                for name in files:
                    os.remove(osp.join(walk_root, name))
                for name in dirs:
                    os.rmdir(osp.join(walk_root, name))
            os.rmdir(self._root)
        _write_minimal_dataset(self._root)

    def tearDown(self):
        if osp.isdir(self._root):
            for walk_root, dirs, files in os.walk(self._root, topdown=False):
                for name in files:
                    os.remove(osp.join(walk_root, name))
                for name in dirs:
                    os.rmdir(osp.join(walk_root, name))
            os.rmdir(self._root)

    def test_splits_and_tuple_shapes(self):
        ds = ToyCorruption(self._root, verbose=False)
        self.assertEqual(len(ds.train), 3)
        self.assertEqual(len(ds.query_s1), 1)
        self.assertEqual(len(ds.query_s2), 1)
        self.assertEqual(len(ds.gallery), 1)
        self.assertEqual(ds.num_train_pids, 2)
        self.assertEqual(ds.num_train_imgs, 3)

        for row in ds.train:
            self.assertEqual(len(row), 4)
            path, pid, cam, sev = row
            self.assertTrue(osp.isabs(path))
            self.assertTrue(osp.isfile(path))
            self.assertIn(pid, (0, 1))
            self.assertGreaterEqual(sev, 0)
            self.assertLessEqual(sev, 4)

        for row in (ds.query_s1[0], ds.query_s2[0], ds.gallery[0]):
            self.assertEqual(len(row), 3)
            path, pid, cam = row
            self.assertTrue(osp.isfile(path))
            self.assertEqual(pid, 0)
            self.assertIn(cam, (1, 2))

        self.assertIn("query_s1", ds.query_s1[0][0])
        self.assertIn("query_s2", ds.query_s2[0][0])
        self.assertIn("gallery", ds.gallery[0][0])
        train_pids = {r[1] for r in ds.train}
        self.assertEqual(train_pids, {0, 1})

    def test_factory_create(self):
        ds = create("toy_corruption", self._root, verbose=False)
        self.assertIsInstance(ds, ToyCorruption)
        self.assertEqual(ds.num_train_imgs, 3)

    def test_factory_unknown_raises(self):
        with self.assertRaises(KeyError):
            create("not_a_real_dataset", "/tmp")

    def test_train_severity_and_pid_relabel(self):
        ds = ToyCorruption(self._root, verbose=False)
        by_tail = {osp.basename(r[0]): r for r in ds.train}
        self.assertEqual(by_tail["0001_c1s1_000001_01.jpg"][3], 0)
        self.assertEqual(by_tail["0001_c1s2_000001_01.jpg"][3], 1)
        self.assertEqual(by_tail["0002_c1s1_000001_01.jpg"][3], 4)

    def test_missing_metadata_raises(self):
        empty = osp.join(_ROOT, "tests", "_tmp_toy_empty")
        os.makedirs(empty, exist_ok=True)
        try:
            with self.assertRaises(RuntimeError) as ctx:
                ToyCorruption(empty, verbose=False)
            self.assertIn("metadata", str(ctx.exception).lower())
        finally:
            os.rmdir(empty)

    def test_bad_payload_raises(self):
        bad = osp.join(_ROOT, "tests", "_tmp_toy_bad")
        os.makedirs(bad, exist_ok=True)
        try:
            with open(osp.join(bad, "metadata.json"), "w") as f:
                json.dump({"foo": 1}, f)
            with self.assertRaises(ValueError) as ctx:
                ToyCorruption(bad, verbose=False)
            self.assertIn("images", str(ctx.exception).lower())
        finally:
            os.remove(osp.join(bad, "metadata.json"))
            os.rmdir(bad)


_REPO_TOY_ROOT = osp.join(_ROOT, "examples", "data", "ToyCorruption")
_REPO_META = osp.join(_REPO_TOY_ROOT, "metadata.json")


@unittest.skipUnless(osp.isfile(_REPO_META), "ToyCorruption metadata.json not in repo")
class TestToyCorruptionFullMetadata(unittest.TestCase):
    def test_counts_match_metadata(self):
        with open(_REPO_META, "r", encoding="utf-8") as f:
            payload = json.load(f)
        images = payload["images"]
        n_train = sum(1 for e in images.values() if e.get("split") == "train")
        n_eval = sum(1 for e in images.values() if e.get("split") == "eval")
        n_q1 = sum(
            1
            for e in images.values()
            if e.get("split") == "eval"
            and int(e["severity"]) == 0
            and int(e["source_idx"]) == 1
        )
        n_q2 = sum(
            1
            for e in images.values()
            if e.get("split") == "eval"
            and int(e["severity"]) == 0
            and int(e["source_idx"]) == 2
        )
        n_gal = n_eval - n_q1 - n_q2

        ds = ToyCorruption(_REPO_TOY_ROOT, verbose=False)
        self.assertEqual(len(ds.train), n_train)
        self.assertEqual(len(ds.query_s1), n_q1)
        self.assertEqual(len(ds.query_s2), n_q2)
        self.assertEqual(len(ds.gallery), n_gal)
        self.assertEqual(ds.num_train_imgs, n_train)
        self.assertEqual(ds.num_train_pids, len({t[1] for t in ds.train}))
        sevs = {t[3] for t in ds.train}
        self.assertEqual(sevs, {0, 1, 2, 3, 4})

    def test_query_clean_filenames_use_seq1(self):
        ds = ToyCorruption(_REPO_TOY_ROOT, verbose=False)
        for q in (ds.query_s1, ds.query_s2):
            for path, _pid, _cam in q:
                self.assertTrue(
                    path.endswith("s1_000001_01.jpg"),
                    msg="severity 0 => seq in filename is s1",
                )


if __name__ == "__main__":
    unittest.main()
