"""
Contains some commonly used functions b/w all modules.
"""

import torch
import torch.nn as nn

from scratchai import nets


__all__ = ['IntermediateLayer', 'Flatten', 'Debug']


class IntermediateLayer(nn.Module):
  """
  This class implements a Network that helps to get intermediate 
  layers from other Networks.

  Arguments
  ---------
  net           : nn.Module, str
                  The net from which to return the intermediate layers of.
  return_layers : list
                  Whose each element contains the details of the layer
                  from which to return the results.
                  So, for each element in return_layer this module returns
                  a result. And each element in the return_layers is
                  a pattern which indicates which layer to take the output
                  from. 
                  More on what type patterns the return_layers can take:
                  - 'key' : The usual key which comes from `named_modules()`
                  - 'b_x' : Output of the layer which comes just before layer
                            'x'. `x` should be a layer name from torch.nn.
                            Eg: `x` can be `AdaptiveAvgPool2d`
                  
                  Note: For now, it just supports the above defined patterns.

  Returns
  -------
  out : list
        
  """
  def __init__(self, net, return_layers):
    super().__init__()
    assert type(net) in [str, nn.Module]
    if isinstance(net, str): net = getattr(nets, net)()

    self.net = net
    self.return_layers = return_layers
    """
    self.keys = []; p = None
    for key, module in net.named_modules():
      for pattern in return_layers:
        # If `_` is in pattern then its not a key but a pattern
        if '_' in pattern:
          expression, layer = pattern.split('_')

          # Get the layer`required
          layer = getattr(nn, layer)

          if expression == 'b': self.keys.append(p)

        else: self.keys.append(
    """
  
  def forward(self, x):
    out = {}
    for key, layer in self.net.named_parameters():
      x = layer(x)
      if key in self.return_layers: out[key] = x
    return out


class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)


class Debug(nn.Module):
  def __init__(self, debug_value=None):
    super().__init__()
    self.debug_value = debug_value
  def forward(self, x):
    if self.debug_value: print (self.debug_value, end='')
    print (x.shape)
    return x
