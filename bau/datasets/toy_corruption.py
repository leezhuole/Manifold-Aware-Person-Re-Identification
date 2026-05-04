# encoding: utf-8
"""ToyCorruption dataset loader (metadata-driven toy ReID corruption splits)."""

from __future__ import absolute_import, print_function

import json
import os.path as osp

from ..utils.data import BaseImageDataset


class ToyCorruption(BaseImageDataset):
    """Controlled toy corruption dataset from ``metadata.json`` (v4 schema).

    Expects on-disk layout produced by ``scripts/generate_toy_dataset.py``:

    - ``bounding_box_train/*.jpg`` — train split (750 PIDs × 4 sources × 5 severities)
    - ``query_s1/``, ``query_s2/``, ``gallery/`` — eval partition (50 PIDs)

    Parameters
    ----------
    root : str
        Directory containing ``metadata.json`` and the subtrees above
        (e.g. ``examples/data/ToyCorruption``).
    """

    _META_NAME = "metadata.json"

    def __init__(self, root, verbose=True, **kwargs):
        super(ToyCorruption, self).__init__()
        self.dataset_dir = osp.abspath(osp.expanduser(root))
        self._meta_path = osp.join(self.dataset_dir, self._META_NAME)

        self._check_before_run()
        with open(self._meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if "images" not in payload or "dataset" not in payload:
            raise ValueError(
                "metadata.json must contain top-level 'dataset' and 'images' keys"
            )

        train, query_s1, query_s2, gallery, by_source_severity = self._build_lists(
            payload["images"]
        )

        if verbose:
            print("=> ToyCorruption loaded from {}".format(self.dataset_dir))
            self._print_statistics(train, query_s1, query_s2, gallery)

        self.train = train
        self.query_s1 = query_s1
        self.query_s2 = query_s2
        self.gallery = gallery
        self.by_source_severity = by_source_severity

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = (
            self.get_imagedata_info(self.train)
        )

    def _check_before_run(self):
        if not osp.isdir(self.dataset_dir):
            raise RuntimeError("'{}' is not a directory".format(self.dataset_dir))
        if not osp.isfile(self._meta_path):
            raise RuntimeError(
                "ToyCorruption metadata missing: '{}'".format(self._meta_path)
            )

    @staticmethod
    def _basename_from_key(key):
        # Keys are like "train/0001_c1s1_000001_01.jpg" or "eval/0001_c1s1_000001_01.jpg"
        return osp.basename(key.replace("\\", "/"))

    @staticmethod
    def _eval_subdir_and_tuple(basename, entry):
        """Map an eval ``images`` entry to (subdir, (basename, pid0, camid))."""
        severity = int(entry["severity"])
        source_idx = int(entry["source_idx"])
        new_pid = int(entry["new_pid"])
        cam_id = int(entry.get("cam_id", source_idx))
        pid0 = new_pid - 1

        if severity == 0 and source_idx == 1:
            subdir = "query_s1"
        elif severity == 0 and source_idx == 2:
            subdir = "query_s2"
        else:
            subdir = "gallery"

        return subdir, (basename, pid0, cam_id)

    def _build_lists(self, images_map):
        train = []
        query_s1 = []
        query_s2 = []
        gallery = []
        by_source_severity = {}

        for key in sorted(images_map.keys()):
            entry = images_map[key]
            split = entry.get("split", key.split("/", 1)[0])
            basename = self._basename_from_key(key)

            if split == "train":
                rel_train = osp.join("bounding_box_train", basename)
                img_path = osp.join(self.dataset_dir, rel_train)
                new_pid = int(entry["new_pid"])
                cam_id = int(entry.get("cam_id", entry["source_idx"]))
                severity = int(entry["severity"])
                pid0 = new_pid - 1
                train.append((img_path, pid0, cam_id, severity))

            elif split == "eval":
                subdir, triple = self._eval_subdir_and_tuple(basename, entry)
                img_path = osp.join(self.dataset_dir, subdir, triple[0])
                row = (img_path, triple[1], triple[2])
                if subdir == "query_s1":
                    query_s1.append(row)
                elif subdir == "query_s2":
                    query_s2.append(row)
                else:
                    gallery.append(row)

                # Balanced-eval index: unified bounding_box_test/ paths by (source, severity).
                severity = int(entry["severity"])
                source_idx = int(entry["source_idx"])
                bbt_path = osp.join(self.dataset_dir, "bounding_box_test", basename)
                ss_key = (source_idx, severity)
                by_source_severity.setdefault(ss_key, []).append(
                    (bbt_path, triple[1], triple[2])
                )
            else:
                raise ValueError("Unknown split '{}' for key '{}'".format(split, key))

        return train, query_s1, query_s2, gallery, by_source_severity

    def _print_statistics(self, train, query_s1, query_s2, gallery):
        nt_p, nt_i, nt_c = self.get_imagedata_info(train)
        q1_p, q1_i, q1_c = self.get_imagedata_info(query_s1)
        q2_p, q2_i, q2_c = self.get_imagedata_info(query_s2)
        g_p, g_i, g_c = self.get_imagedata_info(gallery)
        print("Dataset statistics:")
        print("  ---------------------------------------------------")
        print("  subset     | # ids | # images | # cameras")
        print("  ---------------------------------------------------")
        print(
            "  train      | {:5d} | {:8d} | {:9d}".format(nt_p, nt_i, nt_c)
        )
        print(
            "  query_s1   | {:5d} | {:8d} | {:9d}".format(q1_p, q1_i, q1_c)
        )
        print(
            "  query_s2   | {:5d} | {:8d} | {:9d}".format(q2_p, q2_i, q2_c)
        )
        print(
            "  gallery    | {:5d} | {:8d} | {:9d}".format(g_p, g_i, g_c)
        )
        print("  ---------------------------------------------------")
