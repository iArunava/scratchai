"""
Contains some commonly used functions b/w all modules.
"""

import torch
import torch.nn as nn

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
