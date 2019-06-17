"""
Contains some commonly used functions b/w all modules.
"""

import torch
import torch.nn as nn

from collections import OrderedDict

class InterLayer(nn.Module):
  def __init__(self, layer, return_layers):
    super().__init__()
    self.layer = layer
    self.return_layers = return_layers
  
  def forward(self, x):
    out = OrderedDict()
    for name, layer in self.layer.named_children():
      x = layer(x)
      if name in self.return_layers:
        out[self.return_layers[name]] = x
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
