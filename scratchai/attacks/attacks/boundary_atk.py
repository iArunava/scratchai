"""
The Boundary Attack - First Effective Decision Based Attack.
Paper: https://arxiv.org/pdf/1712.04248.pdf
"""

import torch

def boundary(x:torch.Tensor=None, net:nn.Module, xt:torch.Tensor):
  """
  Implements the Boundary Attack.
  Brendel & Rauber et al.

  Arguments
  ---------
  net : nn.Module
        The Black Box model.
  x   : torch.Tensor
        The input to image.
  xt  : torch.Tensor, None
        Defaults to None. If not None, attack is targeted and then
        `x` is perturbed in a way so as to misclassify it to the class
        predicted by `xt`.
  """
