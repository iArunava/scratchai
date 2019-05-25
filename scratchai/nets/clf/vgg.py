"""
VGG Model
"""

import torch
import torch.nn as nn
from scratchai.nets.common import Flatten


__all__ = ['VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 
           'vgg16_bn', 'vgg19', 'vgg19_bn']


def conv(ic:int, oc:int, k:int=3, s:int=1, p:int=1, norm:bool=True):
  layers = [nn.Conv2d(ic, oc, k, s, p, bias=not norm)]
  if norm: layers += [nn.BatchNorm2d(oc)]
  layers += [nn.ReLU(inplace=True)]
  return layers

def linear(inn:int, onn:int, drop:bool=True):
  layers = [nn.Linear(inn, onn)]
  if drop: layers += [nn.Dropout()]
  layers += [nn.ReLU(inplace=True)]
  return layers


# TODO Add the VGG A-LRN and VGG C models

class VGG(nn.Module):
  """
  Implementation of VGG model.
  Paper: https://arxiv.org/pdf/1409.1556.pdf

  Arguments
  ---------
  nc : int
       The number of classes
  lconf : list
          The configuration of the convs in the net, this is the conf
          that makes the implementation 11layers or 19layers or more as needed.
          Default: [1, 1, 2, 2, 2] - which is VGG11
  norm : bool
         If true, BatchNormalization is used after each layer.
         Defaults to True.
  """
  def __init__(self, nc=1000, lconf:list=[1, 1, 2, 2, 2], norm:bool=True):
    super().__init__()
    
    ic = 3; oc = 64
    layers = []
    for l in lconf:
      layers += conv(ic, oc, norm=norm)
      ic = oc
      for _ in range(l-1): layers += conv(ic, oc, norm=norm)
      oc *= 2 if oc*2 <= 512 else 1
      layers += [nn.MaxPool2d(2, 2)]
    
    layers += [nn.AdaptiveAvgPool2d((7, 7)), Flatten(), 
               *linear(512 * 7 * 7, 4096), *linear(4096, 4096), 
               nn.Linear(4096, nc)]

    self.net = nn.Sequential(*layers)

  def forward(self, x): return self.net(x)


### VGG A ###
def vgg11():
  return VGG(lconf=[1, 1, 2, 2, 2], norm=False)

def vgg11_bn():
  return VGG(lconf=[1, 1, 2, 2, 2])

### VGG B ###
def vgg13():
  return VGG(lconf=[2, 2, 2, 2, 2], norm=False)

def vgg13_bn():
  return VGG(lconf=[2, 2, 2, 2, 2])

### VGG D ###
def vgg16():
  return VGG(lconf=[2, 2, 2, 2, 2], norm=False)
 
def vgg16_bn():
  return VGG(lconf=[2, 2, 2, 2, 2])

### VGG E ###
def vgg19():
  return VGG(lconf=[2, 2, 4, 4, 4], norm=False)
  
def vgg19_bn():
  return VGG(lconf=[2, 2, 4, 4, 4])
