"""
The Noise Attack
"""

import numpy as np
import torch

from torchvision import transforms as T
from torchvision import datasets

__all__ = ['noise', 'Noise']

def noise(x, eps=0.3, order=np.inf, clip_min=None, clip_max=None):
    """
    A weak attack that just picks a random point in the attacker's action
    space. When combined with an attack bundling function, this can be used to
    implement random search.

    References:
    https://arxiv.org/abs/1802.00420 recommends random search to help identify
        gradient masking

    https://openreview.net/forum?id=H1g0piA9tQ recommends using noise as part
        of an attack building recipe combining many different optimizers to
        yield a strong optimizer.

    Arguments
    ---------
    x : torch.Tensor
        The input image.
    """
    
    if order != np.inf: raise NotImplementedError(ord)
    
    eta = torch.FloatTensor(*x.shape).uniform_(-eps, eps).to(x.device)
    adv_x = x + eta

    if clip_min is not None and clip_max is not None:
        adv_x = torch.clamp(adv_x, min=clip_min, max=clip_max)

    return adv_x


###################################################################
# A class to initialize the noise attack
# This class is implemented mainly so this attack can be directly 
# used along with torchvision.transforms

class Noise():
  def __init__(self, net=None, **kwargs):
    self.kwargs = kwargs
  def __call__(self, x):
    return noise(x, **self.kwargs)
