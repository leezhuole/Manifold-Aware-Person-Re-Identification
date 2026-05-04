from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.
    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid, dsetid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        if batch_size < num_instances:
            raise ValueError(
                'batch_size={} must be no less '
                'than num_instances={}'.format(batch_size, num_instances)
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, items in enumerate(data_source):
            pid = items[1]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        assert len(self.pids) >= self.num_pids_per_batch

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class SeverityStratifiedSampler(Sampler):
    """P×K identity sampling with a per-PID severity diversity constraint.

    Same outer schedule as :class:`RandomIdentitySampler`, but each block of
    ``num_instances`` indices for one identity is chosen so that the block
    spans **at least two distinct severity** labels (read from
    ``data_source[i][3]``). Training tuples must be 4-tuples
    ``(path, pid, camid, severity)``; shorter tuples are treated as having a
    single severity bucket (falls back to uniform random sampling for that
    PID).

    Args:
        data_source (list): 4-tuples ``(img_path, pid, camid, severity)``.
        batch_size (int): Total batch size (must be divisible by P×K layout).
        num_instances (int): K — instances per identity per batch (≥2 typical).
    """

    def __init__(self, data_source, batch_size, num_instances):
        if batch_size < num_instances:
            raise ValueError(
                "batch_size={} must be no less "
                "than num_instances={}".format(batch_size, num_instances)
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        self.pid_severity_index = defaultdict(lambda: defaultdict(list))

        for index, items in enumerate(data_source):
            pid = items[1]
            self.index_dic[pid].append(index)
            if len(items) >= 4:
                sev = items[3]
                self.pid_severity_index[pid][sev].append(index)

        self.pids = list(self.index_dic.keys())
        assert len(self.pids) >= self.num_pids_per_batch

        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        # Repairs / within-chunk shuffles must not advance the global RNG used by
        # ``RandomIdentitySampler`` so that P×K interleaving matches bit-for-bit.
        self._aux_rng = random.Random(0)

    @staticmethod
    def _severity_for_item(items):
        if len(items) >= 4:
            return items[3]
        return None

    def _severity_set(self, indices):
        return {
            self._severity_for_item(self.data_source[i]) for i in indices
        }

    def _chunk_has_two_severities(self, chunk):
        return len(self._severity_set(chunk)) >= 2

    def _repair_chunks_for_severity(self, chunks):
        """Swap entries across chunks so each chunk spans ≥2 severities when possible."""
        chunks = [list(c) for c in chunks]
        if not chunks:
            return chunks

        def ok(ch):
            return self._chunk_has_two_severities(ch)

        for ci in range(len(chunks)):
            if ok(chunks[ci]):
                continue
            fixed = False
            for cj in range(len(chunks)):
                if ci == cj:
                    continue
                for ai in range(len(chunks[ci])):
                    for aj in range(len(chunks[cj])):
                        ni, nj = list(chunks[ci]), list(chunks[cj])
                        ni[ai], nj[aj] = nj[aj], ni[ai]
                        if ok(ni) and ok(nj):
                            chunks[ci], chunks[cj] = ni, nj
                            fixed = True
                            break
                    if fixed:
                        break
                if fixed:
                    break
        for ch in chunks:
            self._aux_rng.shuffle(ch)
        return chunks

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True
                ).tolist()
            random.shuffle(idxs)
            batch_idxs = []
            pid_chunks = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    pid_chunks.append(batch_idxs)
                    batch_idxs = []
            for ch in self._repair_chunks_for_severity(pid_chunks):
                batch_idxs_dict[pid].append(ch)

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam) in enumerate(data_source):
            if (pid<0): continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            _, i_pid, i_cam = self.data_source[i]

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = No_index(index, i)
                if (not select_indexes): continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])

        return iter(ret)