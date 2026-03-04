# encoding: utf-8
"""
AG-ReID.v2 dataset loader for the BAU framework.

Supports cross-view experiments via text protocol files:
  - aerial_to_cctv:     query = aerial (C0), gallery = CCTV (C3)
  - aerial_to_wearable: query = aerial (C0), gallery = wearable (C2)
  - cctv_to_aerial:     query = CCTV (C3),   gallery = aerial (C0)
  - wearable_to_aerial: query = wearable (C2), gallery = aerial (C0)

Camera encoding in filenames: C0 = UAV (aerial), C2 = Wearable, C3 = CCTV.

Reference:
  Nguyen et al., "AG-ReID.v2: Bridging Aerial and Ground Views for Person
  Re-Identification", IEEE TIFS 2024.
"""

import glob
import re
import os.path as osp

from ..utils.data import BaseImageDataset

# Map from experiment name to protocol text file
_EXPERIMENT_FILES = {
    'aerial_to_cctv':     'exp1_aerial_to_cctv.txt',
    'aerial_to_wearable': 'exp2_aerial_to_wearable.txt',
    'cctv_to_aerial':     'exp4_cctv_to_aerial.txt',
    'wearable_to_aerial': 'exp5_wearable_to_aerial.txt',
}

# Map from view name to camera IDs
# C0 = UAV (aerial, ~15-45m), C2 = Wearable (~1.5m), C3 = CCTV (~3m)
_VIEW_CAMERA_MAP = {
    'aerial':   [0],       # UAV only
    'wearable': [2],       # Wearable camera only
    'cctv':     [3],       # CCTV only
    'ground':   [2, 3],    # Both ground-level cameras
    'all':      [0, 2, 3], # All cameras (no filtering)
}


class AG_ReID_v2(BaseImageDataset):
    """
    AG-ReID.v2: Aerial-Ground Person Re-Identification dataset.

    Expected directory layout under ``root``::

        root/
        └── AG-ReID.v2/
            ├── AG-ReID.v2/
            │   ├── train_all/  (807 identities)
            │   ├── query/      (808 identities)
            │   └── gallery/    (808 identities)
            ├── qut_attribute_v8.mat
            ├── exp1_aerial_to_cctv.txt
            ├── exp2_aerial_to_wearable.txt
            ├── exp4_cctv_to_aerial.txt
            └── exp5_wearable_to_aerial.txt

    Parameters
    ----------
    root : str
        Path to the data directory (``--data-dir``).
    experiment : str or None
        One of ``'aerial_to_cctv'``, ``'aerial_to_wearable'``,
        ``'cctv_to_aerial'``, ``'wearable_to_aerial'``, or ``None``
        (default uses the full query/gallery without camera filtering).
    verbose : bool
        Print dataset statistics.
    """

    dataset_dir = 'AG-ReIDv2'

    def __init__(self, root, experiment=None, view_filter=None, verbose=True, **kwargs):
        super(AG_ReID_v2, self).__init__()
        self.root = root
        # root/AG-ReID.v2/ contains experiment files and a nested AG-ReID.v2/ with images
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.data_dir = osp.join(self.dataset_dir, 'AG-ReID.v2')
        self.train_dir = osp.join(self.data_dir, 'train_all')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')
        self.experiment = experiment
        self.view_filter = view_filter

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)

        # Filter training data by camera view if requested
        if view_filter is not None:
            if view_filter not in _VIEW_CAMERA_MAP:
                raise ValueError(
                    f"view_filter must be one of {list(_VIEW_CAMERA_MAP.keys())}, "
                    f"got '{view_filter}'"
                )
            target_cams = set(_VIEW_CAMERA_MAP[view_filter])
            train = [item for item in train if item[2] in target_cams]
            # Re-label PIDs to be contiguous after filtering
            train = self._relabel_filtered(train)

        if experiment is not None:
            query, gallery = self._load_experiment(experiment)
        else:
            query = self._process_dir(self.query_dir, relabel=False)
            gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            exp_label = experiment if experiment else 'all'
            view_label = view_filter if view_filter else 'all'
            print("=> AG-ReID.v2 loaded (experiment: {}, view: {})".format(exp_label, view_label))
            if view_filter is not None:
                # View-filtered datasets are training-only; query/gallery are
                # inherited from the full dataset and not meaningful here.
                num_pids, num_imgs, num_cams = self.get_imagedata_info(train)
                print("  ----------------------------------------")
                print("  {:8s} | {:5d} | {:8d} | {:9d}".format(
                    'train', num_pids, num_imgs, num_cams))
                print("  ----------------------------------------")
            else:
                self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all required files and directories are available."""
        for path in [self.data_dir, self.train_dir, self.query_dir, self.gallery_dir]:
            if not osp.exists(path):
                raise RuntimeError("'{}' is not available".format(path))
        if self.experiment is not None:
            exp_file = osp.join(self.dataset_dir, _EXPERIMENT_FILES[self.experiment])
            if not osp.exists(exp_file):
                raise RuntimeError("Experiment file '{}' is not available".format(exp_file))

    @staticmethod
    def _relabel_filtered(dataset):
        """Re-label PIDs to be contiguous [0, N) after filtering."""
        pids = sorted(set(item[1] for item in dataset))
        pid2label = {pid: label for label, pid in enumerate(pids)}
        return [(img_path, pid2label[pid], camid) for img_path, pid, camid in dataset]

    def _process_dir(self, dir_path, relabel=False):
        """Scan *dir_path* recursively for ``*.jpg`` images.

        Returns a list of 3-tuples ``(img_path, pid, camid)``.
        """
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'), recursive=True)
        pattern_pid = re.compile(r'P(\d+)')
        pattern_camid = re.compile(r'C(\d+)')

        pid_container = set()
        for img_path in img_paths:
            fname = osp.basename(img_path)
            pid = int(pattern_pid.search(fname).group(1))
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in img_paths:
            fname = osp.basename(img_path)
            pid = int(pattern_pid.search(fname).group(1))
            camid = int(pattern_camid.search(fname).group(1))
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

    def _load_experiment(self, experiment):
        """Load query and gallery splits from an experiment text file.

        Each line is a relative path like
        ``query/P0313T02220A0/P0313T02220A0C0F11011.jpg`` or
        ``gallery/P1048T04071A1/P1048T04071A1C3F06811.jpg``.

        Returns ``(query_list, gallery_list)`` of 3-tuples.
        """
        exp_file = osp.join(self.dataset_dir, _EXPERIMENT_FILES[experiment])
        pattern_pid = re.compile(r'P(\d+)')
        pattern_camid = re.compile(r'C(\d+)')

        query, gallery = [], []
        with open(exp_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path = osp.join(self.data_dir, line)
                fname = osp.basename(line)
                pid = int(pattern_pid.search(fname).group(1))
                camid = int(pattern_camid.search(fname).group(1))

                if line.startswith('query/'):
                    query.append((img_path, pid, camid))
                elif line.startswith('gallery/'):
                    gallery.append((img_path, pid, camid))

        return query, gallery