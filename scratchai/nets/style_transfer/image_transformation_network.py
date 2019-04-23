"""
Image Transformation Network from Justin Johnson et al. 2016. 
"""

import torch
import torch.nn as nn

from torchvision.transforms import CenterCrop

from scratchai.nets.clf.resnet import conv
from scratchai.nets.seg.enet import uconv
from scratchai.nets.debug import Debug

class resblock(nn.Module):
  """
  This module is the implementation of a resnet block.

  Arguments
  ---------
  ic : int
       # of input channels
  oc : int
       # of output channels
  norm : nn.Module
         The Normalization to be used.
  """

  def __init__(self, ic:int, oc:int=None, norm:nn.Module=nn.InstanceNorm2d):
    super().__init__()
    self.main = nn.Sequential(*conv(ic, oc, 3, 1, 0, norm=norm), 
                              *conv(oc, oc, 3, 1, 0, act=None, norm=norm))
    self.act = nn.ReLU(inplace=True)
    
  def forward(self, x):
    return self.act(self.main(x) +  x[:, :, 2:-2, 2:-2])

class ITN(nn.Module):
  """
  This module implements the Image Transformation Network as proposed in the
  Perceptual Losses paper by Justin et al. 2016

  Paper Link: 
  https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf

  Arguments
  ---------
  """

  def __init__(self):
    super().__init__()

    layers = [nn.ReflectionPad2d(40), *conv(3, 32, 9, 1, 4), 
              *conv(32, 64, 3, 2, 1), *conv(64, 128, 3, 2, 1)]

    layers += [resblock(128, 128, norm=nn.InstanceNorm2d) for _ in range(5)]

    layers += [*uconv(128, 64, 4, 2, 1), *uconv(64, 32, 4, 2, 1), 
               *uconv(32, 3, 9, 1, 4)]

    self.net = nn.Sequential(*layers)

  def forward(self, x): return self.net(x)
