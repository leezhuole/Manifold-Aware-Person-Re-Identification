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


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam, _) in enumerate(data_source):
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

            _, i_pid, i_cam, _ = self.data_source[i]

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


class RandomDomainBalancedSampler(Sampler):
    """Sample batches with fixed number of nuisance groups and instances per group.

    Each output step contributes ``num_groups * instances_per_group`` indices (one batch).
    ``group_by='camera'`` uses camera id (tuple index 2); ``group_by='dataset'`` uses
    dataset id (tuple index 3), requiring 4-tuple entries in ``data_source``.
    """

    def __init__(self, data_source, batch_size, instances_per_group, group_by='camera'):
        if batch_size % instances_per_group != 0:
            raise ValueError(
                'batch_size must be divisible by instances_per_group, got {} and {}'.format(
                    batch_size, instances_per_group
                )
            )
        self.data_source = data_source
        self.batch_size = batch_size
        self.instances_per_group = instances_per_group
        self.num_groups = batch_size // instances_per_group
        self.group_by = group_by
        label_idx = 2 if group_by == 'camera' else 3
        self.label_to_indices = defaultdict(list)
        for index, items in enumerate(data_source):
            if group_by == 'dataset' and len(items) < 4:
                raise ValueError(
                    'RandomDomainBalancedSampler with group_by=dataset requires 4-tuple train entries'
                )
            lab = items[label_idx]
            self.label_to_indices[lab].append(index)
        self.labels = list(self.label_to_indices.keys())
        if len(self.labels) < self.num_groups:
            raise ValueError(
                'Not enough distinct {} labels ({}); need >= num_groups={}'.format(
                    group_by, len(self.labels), self.num_groups
                )
            )
        num_batches = max(1, len(data_source) // batch_size)
        self.length = num_batches * batch_size

    def __len__(self):
        return self.length

    def __iter__(self):
        n_batches = self.length // self.batch_size
        final_idxs = []
        for _ in range(n_batches):
            chosen = random.sample(self.labels, self.num_groups)
            for lab in chosen:
                idxs = self.label_to_indices[lab]
                replace = len(idxs) < self.instances_per_group
                picks = np.random.choice(
                    idxs, size=self.instances_per_group, replace=replace
                )
                final_idxs.extend(int(x) for x in picks.tolist())
        return iter(final_idxs)