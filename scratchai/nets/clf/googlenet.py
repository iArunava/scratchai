"""
The GoogLeNet Architecture.
Paper: https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
"""

import torch
import torch.nn as nn


def conv(ic:int, oc:int, k, s, p):
  layers = []
  layers += [nn.Conv2d(ic, oc, k, s, p, bias=False)]
  layers += [nn.BatchNorm2d(oc)]
  layers += [nn.ReLU(inplace=True)]
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

    self.conv1x1 = nn.Conv2d(ic, oc, 1, 1, 0)
    self.conv3x3 = nn.Conv2d

