"""
The LeNet.
Paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
"""

import torch
import torch.nn as nn

from scratchai.nets.common import Flatten

def conv(ic:int, oc:int, k:int=5):
  layers = [nn.Conv2d(ic, oc, kernel_size=5), nn.MaxPool2d(2), 
            nn.ReLU(inplace=True)]
  return layers

class Lenet(nn.Module):
  """
  Implements the Lenet module.
  Paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

  Arguments
  ---------
  nc : int
       # of classes.
  ex : int
       Expansion factor
  """
  def __init__(self, nc:int=10, ex:int=1):
    super().__init__()

    layers = [*conv(3, 6*ex), *conv(6*ex, 16*ex), Flatten(), 
              nn.Linear(16*5*5*ex, 120*ex), nn.Linear(120*ex, 84*ex), 
              nn.Linear(84*ex, nc)]
    self.net = nn.Sequential(*layers)

  def forward(self, x): return self.net(x)
