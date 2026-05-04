from __future__ import absolute_import

from .model import *
from .memory import *

def toy_resnet50(**kwargs):
    """ResNet-50 with θ head for ToyCorruption / L_mono experiments (PLAN.md Step 7)."""
    kwargs.setdefault('with_theta_head', True)
    kwargs.setdefault('num_classes', 0)
    kwargs.setdefault('pretrained', True)
    return resnet50(**kwargs)


__factory = {
    'resnet50': resnet50,
    'toy_resnet50': toy_resnet50,
    'mobilenetv2': mobilenetv2,
    'vitbase': vit_base_patch16,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'resnet50', 'mobilenetv2', 'vitbase'.
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    pretrained : bool, optional
        If True, will load imagenet pre-trained weights. Default: True
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)