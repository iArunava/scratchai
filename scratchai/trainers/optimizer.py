"""
Class to help build Optimizers Easily.
"""

import torch
import torch.nn as nn
import torch.optim as optim

class Optimizer():
  """
  Base Class to build optimizers.

  Arguments
  ---------
  opt : torch.opt
          The optimizer which to use.

  net : nn.Module
          The net whose parameters needs to be optimized.

  lr  : float
          The learning rate.
  
  nesterov : bool
            Whether to enable Nesterov Momentum.

  momentum : float
             The momentum to use (for optimizers that takes in momentums)

  bias_2lr : bool
             Whether to double the learning rates for the biases

  Notes
  -----
  Currently, this optimizer building class supports 2 modes.

  1. It applies the same learning rate to all the parameters (bias_2lr = False)
  2. It applies double learning rates to all the biases (bias_2lr = True)
  """

  def __init__(self, opt:torch.optim, net:nn.Module, lr=1e-3, momentum=0.9,
               nesterov:bool=False, weight_decay:float=5e-4, 
               bias_2lr:bool=False):
    
    assert opt in [optim.SGD, optim.Adam]

    self.lr = lr
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.nesterov = nesterov

    self.bias_2lr = bias_2lr
    
    # Setting the Global Arguments for the optimizer
    optim_global_kwargs = {
      'lr' : self.lr,
      'momentum' : self.momentum,
      'weight_decay' : self.weight_decay,
      'nesterov' : self.nesterov
    }

    if opt == optim.Adam:
      optim_global_kwargs.pop('momentum', None)
      optim_global_kwargs.pop('nesterov', None)

    # Print some info
    if bias_2lr:
      print ('[INFO] Biases will have double learning rate and '
              'weight decay set to zero.')

      self.opt = opt(
        [
         {'params': self.get_parameters(net, bias=False)},
         {'params': self.get_parameters(net, bias=True),
          'lr': self.lr * 2, 'weight_decay': 0},
        ],
        **optim_global_kwargs)
    
    else:
      print ('[INFO] All the parameters will have the same learning rate.')
      self.opt = opt([p for p in net.parameters() if p.requires_grad],
                      **optim_global_kwargs)


  def get_parameters(self, net, bias=False):
    supported_modules = (
      nn.Conv2d,
      nn.ConvTranspose2d,
      nn.Linear
    )

    for m in net.modules():
      if isinstance(m, supported_modules):
        if bias and m.bias is not None and m.bias.requires_grad:
          yield m.bias
        elif m.weight.requires_grad:
          yield m.weight

      else:
        print ('[INFO] Skipping Layer {}'.format(m.__class__.__name__))
        continue
        
  def step(self):
    self.opt.step()

  def zero_grad(self):
    self.opt.zero_grad()

  def state_dict(self):
    return self.opt.state_dict()

  def load_state_dict(self, *args, **kwargs):
    self.opt.load_state_dict(*args, **kwargs)
