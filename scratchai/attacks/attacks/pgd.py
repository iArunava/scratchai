"""
The Projected Gradient Descent attack.
"""

import numpy as np
import torch
import torch.nn as nn
from scratchai.attacks.attacks import fgm
from scratchai.attacks.utils import clip_eta


__all__ = ['pgd', 'PGD']


def pgd(x:torch.Tensor, net:nn.Module, nb_iter:int=10, eps:float=0.3, 
    eps_iter:float=0.05, rand_minmax:float=0.3, clip_min=None, clip_max=None, 
    y=None, ordr=np.inf, rand_init=None, targeted=False) -> torch.Tensor:

  """
  This class implements either the Basic Iterative Method
  (Kuarkin et al. 2016) when rand_init is set to 0. or the
  Madry et al. (2017) method when rand_minmax is larger than 0.
  Paper link (Kuarkin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
  Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
 
  # TODO FIX the below arguments style
  Arguments
  ---------
  model: Model
  dtype: dtype of the data
  default_rand_init: whether to use random initialization by default
  kwargs: passed through to super constructor

  Returns
  -------
  adv_x : torch.Tensor
      The adversarial Example.
  """
  # TODO Check params
  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    asserts.append(x.any() >= clip_min)

  if clip_max is not None:
    asserts.append(x.any() <= clip_max)

  # Initialize loop variables
  if rand_init:
    eta = torch.FloatTensor(*x.shape).uniform_(-minmax, minmax)
  else:
    eta = torch.zeros_like(x)

  # Clip eta
  eta = clip_eta(eta, ordr, eps)
  adv_x = x + eta
  
  if clip_min is not None or clip_max is not None:
    adv_x = torch.clamp(adv_x, clip_max, clip_min)
  
  if y is None:
    # Use ground truth labels to avoid label leaking
    if len(x.shape) == 3: x.unsqueeze_(0)
    _, y = torch.max(net(x), dim=1)
  else:
    targeted = True
 
  if ord == 1:
    raise NotImplementedError("It's not clear that FGM is a good inner loop"
                 " step for PGD when ord=1, because ord=1 FGM "
                 " changes only one pixel at a time. We need "
                 " to rigoursly test a strong ord=1 PGD "
                 " before enabling this feature.")
  i = 0
  while i < nb_iter:
    """
    Do a projected gradient step.
    """
    adv_x = fgm(adv_x, net, eps=eps_iter, ordr=ordr, clip_min=clip_min,
                clip_max=clip_max, y=y, targeted=targeted)

    # Clipping perturbation eta to ord norm ball
    eta = adv_x - x
    eta = clip_eta(eta, ordr, eps)
    adv_x = x + eta

    # Redo the clipping.
    # FGM alread already did it, but subtracting and re-adding eta can add some
    # small numerical error
    if clip_min is not None or clip_max is not None:
      adv_x = torch.clamp(adv_x, clip_min, clip_max)
    i += 1

  # Asserts run only on CPU
  # When multi-GPU eval code tries to force all PGD ops onto GPU, this
  # can cause an error.
  # The 1e-6 is needed to compensate for numerical error.
  # Without the 1e-6 this fails when e.g. eps=.2 clip_max=.5 clip_min=.7
  if ordr == np.inf and clip_min is not None:
    assert (eps <= (1e6 + clip_max - clip_min)) 
  
  return adv_x


##########################################################################
###### Class to initialize the attack for use with torchvision.transforms

class PGD():
  def __init__(self, net, **kwargs):
    self.net = net
    self.kwargs = kwargs
  def __call__(self, x):
    return pgd(x, self.net, **self.kwargs)
