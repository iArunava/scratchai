"""
The GoogLeNet Architecture.
Paper: https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scratchai.nets.utils import get_net
from scratchai.nets.common import Flatten


__all__ = ['InceptionB', 'googlenet']


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
                             

class InceptionB(nn.Module):
  def __init__(self, ic:int, oc1x1:int, mc3x3, oc3x3, mc5x5, oc5x5, ocpool):
    super().__init__()

    self.net1x1 = nn.Sequential(*conv(ic, oc1x1, 1, 1, 0))

    self.net3x3 = nn.Sequential(*conv(ic, mc3x3, 1, 1, 0), 
                                *conv(mc3x3, oc3x3, 3, 1, 1))

    self.net5x5 = nn.Sequential(*conv(ic, mc5x5, 1, 1, 0), 
                                *conv(mc5x5, oc5x5, 5, 1, 2))

    self.pool   = nn.Sequential(nn.MaxPool2d(3, 1, 1), 
                                *conv(ic, ocpool, 1, 1, 0))
def forward(self, x):
    x1 = self.net1x1(x)
    x2 = self.net3x3(x)
    x3 = self.net5x5(x)
    x4 = self.pool(x)

    x = torch.cat([x1, x2, x3, x4], dim=1)
    return x
    

class GoogLeNet(nn.Module):
  def __init__(self, nc:int=1000, inception=InceptionB):
    super().__init__()

    self.conv1 = nn.Sequential(*conv(3, 64, 7, 2, 1),
                                nn.MaxPool2d(3, 2))

    self.conv2 = nn.Sequential(*conv(64, 192, 3, 1, 1),
                                nn.MaxPool2d(3, 2))

    self.inception3a = inception(192, 64, 96, 128, 16, 32, 32)
    self.inception3b = inception(256, 128, 128, 192, 32, 96, 64)

    self.inception4a = inception(480, 192, 96, 208, 16, 48, 64)
    self.inception4b = inception(512, 160, 112, 224, 24, 64, 64)
    self.inception4c = inception(512, 128, 128, 256, 24, 64, 64)
    self.inception4d = inception(512, 112, 144, 288, 32,  64, 64)
    self.inception4e = inception(528, 256, 160, 320, 32, 128, 128)

    self.inception5a = inception(832, 256, 160, 320, 32, 128, 128)
    self.inception5b = inception(832, 384, 192, 384, 48, 128, 128)
    
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1024, nc))
    
    self.maxpool = nn.MaxPool2d(3, 1)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)

    x = self.inception3a(x)
    x = self.inception3b(x)
    
    x = self.maxpool(x)

    x = self.inception4a(x)
    x = self.inception4b(x)
    x = self.inception4c(x)
    x = self.inception4d(x)
    x = self.inception4e(x)

    x = self.maxpool(x)

    x = self.inception5a(x)
    x = self.inception5b(x)
    
    x = self.avgpool(x)
    x = Flatten()(x)
    x = self.classifier(x)
    return x


def googlenet(pretrained=False, **kwargs):
  return get_net(GoogLeNet, pretrained=pretrained, pretrain_url=None, 
                 fname='googlenet', kwargs_net=kwargs, attr='classifier',
                 inn=1024)
