"""
DeepFool Attack from https://arxiv.org/pdf/1511.04599.pdf 
by Seyed Mohsen et al.
"""

import torch
import torch.nn as nn
import numpy as np
import copy

from torch.autograd.gradcheck import zero_gradients as zgrad


__all__ = ['deepfool', 'DeepFool']


def deepfool(x, net:nn.Module, eta:float=0.02, tnc:int=10, miter:int=50, 
             mret=False):
  """
  Implementation of the DeepFool Attack.

  Arguments
  ---------
  x : torch.Tensor
      The input tensor
  net : nn.Module
        The net.
  eta : float
        Eta as used in the paper.
  tnc : int
       The number of classes to consider.
  miter : int
          The maximum number of iterations.
  mret : bool
         If True returns the minimal perturbation that was added 
         to the original image. Defaults to False.
  """
  # Set net to eval
  net.eval()
  dev = x.device

  # If x doesn't have a batch dimension add 1
  # The loop runs twice to ensure that 2D images gets the batch dim
  # For 2D images, 1st iter adds -> The channel dim
  # 2nd iter adds -> batch dim
  for _ in range(2): 
    if len(x.shape) < 3: x = x.unsqueeze(0)
  bs = x.shape[0]
  lbs = list(range(bs))
  
  # x_idt is the initial advesarial image
  x_idt = x.detach().clone().requires_grad_(True)
  # Get the logits (x_idt is not perturbed yet, the logits are true logits)
  logits = net(x_idt)
  # Get the top tnc classes that the net is most confident about
  psort = torch.argsort(logits.data.cpu(), dim=1, descending=True)
  psort = psort[:, :tnc].data.numpy()
  
  tlabl = psort[:, 0] # True Label
  plabl = tlabl       # Pert label

  x_shape = x_idt.shape  # [N x C x H x W]
  w = np.zeros(x_shape)  # [N x C x H x W]
  rt = np.zeros(x_shape) # [N x C x H x W]
  
  i = 0
  while (plabl == tlabl).any() and i < miter:
    # Initial Perturbation
    pert = 5e10
    logits[lbs, tlabl].sum().backward(retain_graph=True)
    ograd = x_idt.grad.data.cpu().numpy().copy()

    for c in range(1, tnc):
      zgrad(x_idt)
      logits[lbs, psort[:, c]].sum().backward(retain_graph=True)
      cgrad = x_idt.grad.data.cpu().numpy().copy()

      # Get new wk and fk
      wk = cgrad - ograd
      fk = (logits[lbs, psort[:, c]] - logits[lbs, tlabl]) \
             .detach().cpu().data.numpy()

      cpert = abs(fk) / np.linalg.norm(wk.reshape(bs, -1), axis=1)
      blist = (cpert < pert).astype(np.float32)
      if blist.any(): 
        pert = pert*(1-blist) + (cpert*blist)
        w = (w.T*(1-blist) + (wk.T * blist)).T
    
    # Added 1e-4 for numerical stability
    ri =  (((pert+1e-4) * w.T) / np.linalg.norm(w.reshape(bs, -1), axis=1)).T
    rt += ri
    
    x_idt = x + ((1+eta) * torch.from_numpy(rt)).float().to(dev)
    x_idt.requires_grad_(True); logits = net(x_idt)
    plabl = torch.argmax(logits, dim=1).detach().cpu().data.numpy()

    i += 1
  
  rt = (1+eta) * rt
  return (x_idt, rt) if mret else x_idt

##################################################################

class DeepFool():
  def __init__(self, net, **kwargs):
    self.net = net
    self.kwargs = kwargs
  def __call__(self, x):
    return deepfool(x, self.net, **self.kwargs)
