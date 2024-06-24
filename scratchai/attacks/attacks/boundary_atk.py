"""
The Boundary Attack - First Effective Decision Based Attack.
Paper: https://arxiv.org/pdf/1712.04248.pdf
"""

import torch
import torch.nn as nn


__all__ = ['boundary']


def boundary(x:torch.Tensor, net:nn.Module, xt:torch.Tensor=None, miter=50,
             eps:float=0.3):
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
  n_calls = 0

  target = True
  if xt is None: xt = torch.rand_like(xt); target = False
  
  # Assert the image passed is normalized
  assert x.max() <= 10. and x.min() >= -10.
  assert xt.max() <= 10. and xt.min() >= -10.
  
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
    print ('f')
    # TODO Check if this loop is really needed
    for _ in range(10):
      xt = torch.rand_like(x)
      plabl = torch.argmax(net(xt.unsqueeze(0)), dim=1)
      if tlabl != plabl: break
  
  # =======================================
  # Reach the boundary in the first step
  # =======================================

  while 1:
    #curr_xt = xt + pert_to_boundary(eps * get_l2(x, xt), x, xt)
    curr_xt = xt + pert_to_boundary2(eps, x, xt)
    curr_labl = torch.argmax(net(xt), dim=1); n_calls += 1

    if curr_labl != plabl: break
    else: xt = curr_xt
  
  # Seems to be working till here
  # TODO Remove
  return xt

  k = 0
  while k <= miter:
   curr_x = xt + pert_from_proposal_distribution()
   curr_labl = torch.argmax(net(curr_x), dim=1)

   if curr_labl != tlabl: xt = curr_x

   k = -~k

  return xt


def get_l2(x, xt):
  diff = []
  for ii, channel in enumerate(x):
    diff.append(torch.norm(xt - x))
  return torch.Tensor(diff)

def pert_to_boundary(eps, x, xt):
  pert = xt - x
  pert /= get_l2(x, xt)
  pert *= eps
  return pert

def pert_to_boundary2(eps, x, xt):
  pert = xt - x
  return pert * eps
