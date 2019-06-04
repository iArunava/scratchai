"""
The Boundary Attack - First Effective Decision Based Attack.
Paper: https://arxiv.org/pdf/1712.04248.pdf
"""

import torch
import torch.nn as nn

def boundary(x:torch.Tensor, net:nn.Module, xt:torch.Tensor=None, miter=50):
  """
  Implements the Boundary Attack.
  Brendel & Rauber et al.

  Arguments
  ---------
  net : nn.Module
        The Black Box model.
  x   : torch.Tensor, None
        The input to image.
  xt  : torch.Tensor, None
        Defaults to None. If not None, attack is targeted and then
        `x` is perturbed in a way so as to misclassify it to the class
        predicted by `xt`.
  """
  target = True
  if xt is None: xt = torch.rand_like(xt); target = False
  
  # Assert the image passed is normalized
  assert x.max() <= 1. and x.min() >= 0.
  assert xt.max() <= 1. and xt.min() >= 0.
  
  if len(x.shape) == 3: x = x.unsqueeze(0)
  if len(xt.shape) == 3: xt = xt.unsqueeze(0)

  # Assert `xt` has a different label from `x`
  net.eval()
  tlabl = torch.argmax(net(x), dim=1)
  plabl = torch.argmax(net(xt), dim=1)
  # Raise Error if the attack is untargeted
  if target: assert tlabl != plabl, 'Adversarial and True labels are equal!'
  # Try to generate a initial (noisy) "adversarial" image.
  else:
    # TODO Check if this loop is really needed
    for _ in range(10):
      xt = torch.rand_like(x)
      plabl = torch.argmax(net(xt.unsqueeze(0)), dim=1)
      if tlabl != plabl: break
  
  k = 0
  while k <= miter:
   curr_x = xt + pert_from_proposal_distribution()
   curr_labl = torch.argmax(net(curr_x), dim=1)

   if curr_labl != tlabl: xt = curr_x

   k = -~k

  return xt
