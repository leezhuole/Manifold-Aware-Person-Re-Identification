from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss
from .crossentropy import CrossEntropyLabelSmooth
from .mono import MonotonicityLoss, count_monotonicity_pairs

__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'MonotonicityLoss',
    'count_monotonicity_pairs',
]
