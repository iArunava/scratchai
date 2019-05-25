"""
The Fast Gradient Method attack.
"""

import numpy as np
import torch
import torch.nn as nn

from scratchai.attacks.utils import optimize_linear


__all__ = ['fgm', 'FGM']

def fgm(x, net:nn.Module, eps:float=0.3, ordr=np.inf, y=None, 
        clip_min=None, clip_max=None, targeted=False, sanity_checks=True):
  """
  Implementation of the Fast Gradient Method.

  Arguments
  ---------
  net : nn.Module
        The model on which to perform the attack.
  x : torch.tensor
      The input to the net.
  y : torch.tensor, optional
      True labels. If targeted is true, then provide the target label.
      Otherwise, only model predictions are used as labels to avoid 
      the "label leaking" effect (explained in this paper: 
      https://arxiv.org/abs/1611.01236). Default is None.
      Labels should be one hot encoded.
  eps: float, optional
       The epsilon (input variation parameter)
  ord : [np.inf, 1, 2], optional
        Order of the norm.
  clip_min : float
             Minimum float value for adversarial example components.
  clip_max : float
             Maximum float value for adversarial example components.
  targeted : bool, optional
             Is the attack targeted or untargeted? Untargeted will try to make
             the label incorrect. Targeted will instead try to move in the
             direction of being more like y.

  Returns
  -------
  adv_x : torch.Tensor
          The adversarial example.
  """
  if ordr not in [np.inf, 1, 2]:
    raise ValueError('Norm order must be either np.inf, 1, or 2.')

  if clip_min:
    assert torch.all(x > torch.tensor(clip_min, device=x.device, dtype=x.dtype))
  if clip_max:
    assert torch.all(x < torch.tensor(clip_max, device=x.device, dtype=x.dtype))
  
  # Flag to indicate if the image has been unsqueezed
  usq = False
  # Inplace operations not working for some bug #15070
  # TODO Update when fixed
  if len(x.shape) == 3: x = x.unsqueeze(0); usq = True

  # x needs to have requires_grad set to True 
  # for its grad to be computed and stored properly in a backward call
  x = x.detach().clone(); x.requires_grad_(True)

  if y is None: _, y = torch.max(net(x), dim=1)

  # Compute loss
  crit = nn.CrossEntropyLoss()
  loss = crit(net(x), y)
  # If attack is targeted, minimize loss of target label rather than maximize
  # loss of correct label.
  if targeted: loss = -loss

  # Define gradient of loss wrt input
  loss.backward()
  opt_pert = optimize_linear(x.grad, eps, ordr)

  # Add perturbation to original example to obtain adversarial example
  adv_x = x + opt_pert

  # If clipping is needed, reset all values outside of [clip_min, clip_max]
  if (clip_min is not None) or (clip_max is not None):
    # We don't currently support one-sided clipping
    assert clip_min is not None and clip_max is not None
    adv_x = torch.clamp(adv_x, clip_min, clip_max)
  
  return adv_x if not usq else adv_x.squeeze()

###################################################################
# A class to initialize the attack
# This class is implemented mainly so this attack can be directly
# used along with torchvision.transforms

class FGM():
  def __init__(self, net, **kwargs):
    self.net = net
    self.kwargs = kwargs
  def __call__(self, x):
    return fgm(x, self.net, **self.kwargs)
