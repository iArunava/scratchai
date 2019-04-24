"""
This is a Debug Layer which helps print debugging outputs after each layer.
"""

import torch
import torch.nn as nn

class Debug(nn.Module):
  def __init__(self, debug_value=None):
    super().__init__()
    self.debug_value = debug_value
  def forward(self, x):
    if self.debug_value: print (self.debug_value, end='')
    print (x.shape)
    return x
