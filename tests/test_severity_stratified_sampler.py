# encoding: utf-8
"""Unit tests for ``SeverityStratifiedSampler`` (PLAN.md Step 2)."""

from __future__ import absolute_import, print_function

import random
import unittest

from bau.utils.data.sampler import RandomIdentitySampler, SeverityStratifiedSampler


def _severity(data_source, idx):
    row = data_source[idx]
    return row[3] if len(row) >= 4 else None


def _pid(data_source, idx):
    return data_source[idx][1]


class TestSeverityStratifiedSampler(unittest.TestCase):
    def _make_pid_block(self, pid, severities):
        """Return list of 4-tuples for one identity (distinct paths)."""
        return [
            ("/p{}_s{}.jpg".format(pid, s), pid, 0, s) for s in severities
        ]

    def test_len_matches_random_identity_sampler(self):
        """Same P×K accounting as ``RandomIdentitySampler`` (pre- and post-``__iter__``)."""
        data = []
        for pid in range(12):
            data.extend(self._make_pid_block(pid, [0, 0, 1, 1, 2, 2, 3, 3]))
        batch_size, k = 16, 4
        s1 = RandomIdentitySampler(data, batch_size, k)
        s2 = SeverityStratifiedSampler(data, batch_size, k)
        self.assertEqual(len(s1), len(s2))
        # Each ``__iter__`` consumes the global RNG; re-seed so both samplers see
        # the same shuffle / interleaving stream as in an isolated training run.
        random.seed(0)
        out1 = list(s1.__iter__())
        random.seed(0)
        out2 = list(s2.__iter__())
        self.assertEqual(len(out1), len(out2))
        self.assertEqual(len(s1), len(out1))
        self.assertEqual(len(s2), len(out2))
        self.assertEqual(len(out1) % batch_size, 0)

    def test_each_pid_chunk_has_at_least_two_severities(self):
        random.seed(1)
        data = []
        for pid in range(16):
            data.extend(self._make_pid_block(pid, [0, 0, 1, 1, 2, 2, 3, 3]))
        batch_size, k = 4, 4
        self.assertEqual(batch_size % k, 0)
        sampler = SeverityStratifiedSampler(data, batch_size, k)
        idxs = list(sampler)
        p_per_batch = batch_size // k
        self.assertEqual(len(idxs) % batch_size, 0)
        for start in range(0, len(idxs), batch_size):
            block = idxs[start : start + batch_size]
            for j in range(p_per_batch):
                chunk = block[j * k : (j + 1) * k]
                p0 = _pid(data, chunk[0])
                for x in chunk:
                    self.assertEqual(_pid(data, x), p0)
                sevs = {_severity(data, x) for x in chunk}
                self.assertGreaterEqual(
                    len(sevs),
                    2,
                    msg="chunk {} sevs={} pids={}".format(
                        chunk, sevs, [_pid(data, x) for x in chunk]
                    ),
                )

    def test_p2_k2_batches_have_pairwise_severity_diversity(self):
        random.seed(2)
        data = []
        for pid in range(20):
            data.extend(self._make_pid_block(pid, [0, 1, 2, 3, 4]))
        batch_size, k = 8, 2
        sampler = SeverityStratifiedSampler(data, batch_size, k)
        idxs = list(sampler)
        for start in range(0, len(idxs), batch_size):
            block = idxs[start : start + batch_size]
            for j in range(batch_size // k):
                a, b = block[j * k], block[j * k + 1]
                self.assertEqual(_pid(data, a), _pid(data, b))
                self.assertNotEqual(_severity(data, a), _severity(data, b))

    def test_short_tuple_falls_back_without_error(self):
        random.seed(3)
        data = [("/a.jpg", 0, 1), ("/b.jpg", 0, 1), ("/c.jpg", 0, 1), ("/d.jpg", 0, 1)]
        sampler = SeverityStratifiedSampler(data, batch_size=4, num_instances=4)
        idxs = list(sampler)
        self.assertEqual(len(idxs), 4)
        self.assertEqual(set(idxs), {0, 1, 2, 3})

    def test_pid_severity_index_populated(self):
        data = self._make_pid_block(7, [0, 1, 2])
        sampler = SeverityStratifiedSampler(data, batch_size=3, num_instances=3)
        self.assertIn(7, sampler.pid_severity_index)
        self.assertEqual(set(sampler.pid_severity_index[7].keys()), {0, 1, 2})
        self.assertEqual(len(sampler.pid_severity_index[7][0]), 1)


if __name__ == "__main__":
    unittest.main()
