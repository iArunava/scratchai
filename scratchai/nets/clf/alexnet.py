"""
Alexnet

https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-
convolutional-neural-networks.pdf
from Alex Krizhevsky Ilya Sutskever Geoffrey E. Hinton
"""

import torch
import torch.nn as nn

from scratchai.nets.common import Flatten


__all__ = ['Alexnet', 'alexnet', 'alexnet_mnist']


def conv(ic:int, oc:int, k:int=3, s:int=1, p:int=1, bk:int=3, bs:int=2):
  layers = [nn.Conv2d(ic, oc, k, s, p), nn.ReLU(inplace=True)]
  layers += [nn.MaxPool2d(bk, bs)]
  return layers

def linear(inn:int, otn:int):
  return [nn.Dropout(), nn.Linear(inn, otn), nn.ReLU(inplace=True)]

class Alexnet(nn.Module):
  """
  Implmentation of Alexnet.
  
  Arguments
  ---------
  nc : int
       # of classes
  ic : int
       # of channels

  References
  ----------
  https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-
  convolutional-neural-networks.pdf
  """
  def __init__(self, nc:int=1000, ic:int=3):
    super().__init__()
    
    layers = [*conv(ic, 64, 11, 4, 2), *conv(64, 192, 5, p=2),
             *conv(192, 384), *conv(384, 256), *conv(256, 256), 
             nn.AdaptiveAvgPool2d((6, 6)), Flatten(), nn.Dropout(),
             *linear(256*6*6, 4096), *linear(4096, 4096), nn.Linear(4096, nc)]
    self.net = nn.Sequential(*layers)
    
  def forward(self, x): return self.net(x)


def alexnet_mnist(pretrained=True, **kwargs):
  kwargs['ic'] = 1
  print ('[INFO] Pretrained network not available!')
  return Alexnet(**kwargs)

def alexnet(pretrained=True, **kwargs):
  print ('[INFO] Pretrained network not available!')
  return Alexnet(**kwargs)
