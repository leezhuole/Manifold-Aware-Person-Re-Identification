from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
import multiprocessing
from PIL import Image
from torch.utils.data import Dataset
from multiprocessing import Manager
import os.path as osp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, cid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, cid


class SeverityPreprocessor(Preprocessor):
    """Preprocessor for 4-tuple toy train rows ``(path, pid, camid, severity)``.

    Returns ``(img_tensor, fname, pid, camid, severity)`` so the training loop can
    feed ``MonotonicityLoss`` and severity-stratified batches.
    """

    def _get_single_item(self, index):
        items = self.dataset[index]
        if len(items) < 4:
            raise ValueError(
                "SeverityPreprocessor expects 4-tuples (path, pid, camid, severity); "
                "got length {}".format(len(items))
            )
        fname, pid, cid, severity = items[0], items[1], items[2], items[3]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, cid, int(severity)


class TwoViewPreprocessor(Dataset):
    def __init__(self, dataset, root=None, transform_w=None, transform_s=None, transform=None):
        super(TwoViewPreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform_w = transform_w
        self.transform_s = transform_s
        
        self.transform = transform # for single view

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, cid, did = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None: # for memory bank init
            img = self.transform(img)
            return img, fname, pid, cid

        img_w = self.transform_w(img)
        img_s = self.transform_s(img)
        return img_w, img_s, pid, did
