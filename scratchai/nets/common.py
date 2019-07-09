"""
Contains some commonly used functions b/w all modules.
"""

import torch
import torch.nn as nn

from collections import OrderedDict

from scratchai import nets


__all__ = ['InterLayer', 'Flatten', 'Debug']


class InterLayer(nn.Module):
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
    #assert type(net) in [str, nn.Module, nn.Sequential]
    if isinstance(net, str): net = getattr(nets, net)()
    self.net = net
    self.return_layers = return_layers
  
  def forward(self, x):
    out = OrderedDict()
    for name, layer in self.net.named_children():
      x = layer(x)
      if name in self.return_layers:
        out[self.return_layers[name]] = x
      
      if len(self.return_layers) == len(out): break
    return out

  def get_ic_for(self, layer_names):
    # TODO Optimize this function. No need to go over all the children.
    out = {}
    for name, layer in self.net.named_children():
      if name in layer_names:
        out[name] = layer.in_channels
      if len(out) == len(layer_names): break
    return out


class Flatten(nn.Module):
  def forward(self, x):
    return x.reshape(x.size(0), -1)


class Debug(nn.Module):
  def __init__(self, debug_value=None):
    super().__init__()
    self.debug_value = debug_value
  def forward(self, x):
    if self.debug_value: print (self.debug_value, end='')
    print (x.shape)
    return x
