"""
The GoogLeNet Architecture.
Paper: https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scratchai.nets.common import Debug


__all__ = ['InceptionA', 'GoogLeNet']


def conv(ic:int, oc:int, k, s, p):
  layers = [nn.Conv2d(ic, oc, k, s, p, bias=False), nn.BatchNorm2d(oc)]
  return layers


class AuxClassifier(nn.Sequential):
  """
  The Auxiliary Classifier as suggested by the paper as added on top of
  4a and 4d modules.

  Arguments
  ---------
  ic : int
       The number of in_channels

  nc : int
       The number of classes to output.
  """
  def __init__(self, ic:int, nc:int=1000):
    super().__init__()

    self.net = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)),
                             nn.Conv2d(ic, 128, 1, 1, 0), Flatten(),
                             nn.Linear(2048, 1024), nn.Dropout2d(0.7),
                             nn.Linear(1024, nc))
                             

class InceptionA(nn.Module):
  def __init__(self, ic:int, oc:int):
    super().__init__()

    self.net1x1 = nn.Sequential(*conv(ic, oc, 1, 1, 0))
    self.net3x3 = nn.Sequential(*conv(ic, oc, 1, 1, 0), *conv(ic, oc, 3, 1, 2))
    self.net5x5 = nn.Sequential(*conv(ic, oc, 1, 1, 0), *conv(ic, oc, 5, 1, 2))
    self.pool   = nn.Sequential(nn.MaxPool2d(ic), *conv(ic, oc, 1, 1, 0))

  def forward(self, x):
    x1 = self.net1x1(x)
    x2 = self.net3x3(x)
    x3 = self.net5x5(x)
    x4 = self.pool(x)

    x = torch.cat([x1, x2, x3, x4], dim=1)
    return x
    

class GoogLeNet(nn.Module):
  def __init__(self, nc:int, inception=InceptionA):
    super().__init__()

    self.conv1 = nn.Sequential(*conv(3, 64, 7, 2, 1), Debug(), 
                                nn.MaxPool2d(3, 2))

    self.conv2 = nn.Sequential(*conv(64, 192, 3, 1, 1), Debug(), 
                                nn.MaxPool2d(3, 2))

    self.inception_3a = inception(192, 256)
    self.inception_3b = inception(256, 480)

  def forward(self, x):
    x = self.conv1(x)
