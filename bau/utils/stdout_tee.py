# encoding: utf-8
"""Tee ``sys.stdout`` to a log file (used by toy training / eval scripts)."""

from __future__ import absolute_import


class StdoutTee(object):
    """Mirror writes to multiple streams (e.g. console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def fileno(self):
        return self.streams[0].fileno()

    def isatty(self):
        return getattr(self.streams[0], "isatty", lambda: False)()
