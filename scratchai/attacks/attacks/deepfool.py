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


# TODO Tests
def deepfool(x, net:nn.Module, eta:float=0.02, nc:int=10, max_iter:int=50):
  # Set net to eval
  net.eval()
  
  # x is the initial advesarial image
  x = x.detach().clone().unsqueeze(0).requires_grad_(True)
  # Get the logits (since x is not perturbed yet, the logits are true logits)
  logits = net(x).data.cpu().numpy().flatten()
  # Get the top nc classes that the net is most confident about
  psort = logits.argsort()[::-1]; psort = psort[:nc]
  # Get the corresponding logits for each predicted class
  llist = [logits[psort[k]] for k in psort]
  
  # TODO Remove print
  print (logits, psort, llist)

  tlabl = psort[0] # True Label
  plabl = tlabl    # Pert label

  x_shape = x.squeeze().shape() # [C x H x W]
  w = np.zeros(x_shape)         # [C x H x W]
  rt = np.zeros(x_shape)     # [C x H x W]
  
  i = 0
  while plabl == tlabl and i < max_iter:
    # Initial Perturbation
    pert = np.inf
    llist[tlabl].backward(retain_graph=True)
    ograd = x.grad.data.cpu().numpy().copy()

    for c in range(1, nc):
      zgrad(x)
      llist[c].backward(retain_graph=True)
      cgrad = x.grad.data.numpy.copy()

      # Get new wk and fk
      wk = cgrad - ograd
      fk = (llist[c] - llist[tlabl]).item().numpy().copy()

      cpert = abs(fk) / np.linalg.norm(wk.flatten())
      if cpert < pert: pert = cpert; w = wk
    
    # Added 1e-4 for numerical stability
    ri =  (pert+1e-4) * w / np.linalg.norm(w.flatten())
    rt += ri
    
    x = x + ri
    llist = net(x).squeeze().data.cpu().numpy().copy().flatten()
    plabl = np.argmax(llist)

  return x

##################################################################

class DeepFool():
  def __init__(self, net, **kwargs):
    self.net = net
    self.kwargs = kwargs
  def __call__(self, x):
    return deepfool(x, self.net, **self.kwargs)
