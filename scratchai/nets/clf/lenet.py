"""
The LeNet.
Paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
"""

import torch
import torch.nn as nn

from scratchai.nets.clf.resnet import conv

def conv(ic:int, oc:int, k:int, s:int, p:int):
  layers = [nn.Conv2d(ic, oc, kernel_size=5), nn.MaxPool2d(2), 
            nn.ReLU(inplace=True)]
  return *layers

class Lenet(nn.Module):
  """
  Implements the Lenet module.
  Paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

  Arguments
  ---------
  mfact : int
  """
  def __init__(self):
    super().__init__()

    self.net = nn.Sequential(*conv(3, 6), *conv(6, 16), nn.Linear(16*5*5, 120),
                             nn.Linear(120, 84), nn.Linear(84, 10))

  def forward(self, x): return self.net(x)
