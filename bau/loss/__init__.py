from __future__ import absolute_import

from .triplet import TripletLoss
from .crossentropy import CrossEntropyLabelSmooth

__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
]
