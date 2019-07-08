"""
The GoogLeNet Architecture.
Paper: https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scratchai.nets.utils import get_net
from scratchai.nets.common import Flatten
from scratchai.pretrained import urls


__all__ = ['InceptionB', 'googlenet', 'googlenet_paper']



def conv(ic:int, oc:int, k, s, p):
  layers = [nn.Conv2d(ic, oc, k, s, p, bias=False), 
            nn.BatchNorm2d(oc, eps=0.001), nn.ReLU(inplace=True)]
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
                             *conv(ic, 128, 1, 1, 0), Flatten(),
                             nn.Linear(2048, 1024), nn.Dropout2d(0.7),
                             nn.Linear(1024, nc))
                             

class InceptionB(nn.Module):
  """
  The Inception Module as described in the GoogLeNet paper.

  Arguements
  ----------
  ic                : int
                      The number of in channels
  
  oc1x1             : int
                      The number of out channels from the 1x1 conv

  mc3x3             : int
                      The number of reduce channels from the 1x1 convs 
                      before getting passed into the 3x3 conv.

  oc3x3             : int
                      The number of out channels from the 3x3 conv

  mc5x5             : int
                      The number of reduce channels from the 1x1 convs 
                      before getting passed into the 5x5 conv.

  oc5x5             : int
                      The number of out channels from the 5x5 conv

  ocpool            : int
                      The number of out channels of the 1x1 conv after 
                      the pooling.

  replace5x5with3x3 : bool
                      Changes the branch with 5x5 convs to 3x3 convs
  """
  def __init__(self, ic:int, oc1x1:int, mc3x3, oc3x3, mc5x5, oc5x5, ocpool,
               replace5x5with3x3:bool=False):
    super().__init__()
    
    ks5x5 = 5
    pad5x5 = 2
    if replace5x5with3x3: 
      ks5x5 = 3
      pad5x5 = 1

    self.net1x1 = nn.Sequential(*conv(ic, oc1x1, 1, 1, 0))

    self.net3x3 = nn.Sequential(*conv(ic, mc3x3, 1, 1, 0), 
                                *conv(mc3x3, oc3x3, 3, 1, 1))

    self.net5x5 = nn.Sequential(*conv(ic, mc5x5, 1, 1, 0), 
                                *conv(mc5x5, oc5x5, ks5x5, 1, pad5x5))

    self.pool   = nn.Sequential(nn.MaxPool2d(3, 1, 1, ceil_mode=True), 
                                *conv(ic, ocpool, 1, 1, 0))

  def forward(self, x):
      x1 = self.net1x1(x)
      x2 = self.net3x3(x)
      x3 = self.net5x5(x)
      x4 = self.pool(x)

      x = torch.cat([x1, x2, x3, x4], dim=1)
      return x
    

class GoogLeNet(nn.Module):
  def __init__(self, nc:int=1000, inception=InceptionB, aux:bool=False,
               **kwargs):
    super().__init__()
    self.aux = aux

    self.conv1 = nn.Sequential(*conv(3, 64, 7, 2, 1),
                                nn.MaxPool2d(3, 2, ceil_mode=True))

    self.conv2 = nn.Sequential(*conv(64, 64, 1, 1, 0), 
                               *conv(64, 192, 3, 1, 1),
                               nn.MaxPool2d(3, 2, ceil_mode=True))

    self.inception3a = inception(192, 64, 96, 128, 16, 32, 32, **kwargs)
    self.inception3b = inception(256, 128, 128, 192, 32, 96, 64, **kwargs)

    self.inception4a = inception(480, 192, 96, 208, 16, 48, 64, **kwargs)
    self.inception4b = inception(512, 160, 112, 224, 24, 64, 64, **kwargs)
    self.inception4c = inception(512, 128, 128, 256, 24, 64, 64, **kwargs)
    self.inception4d = inception(512, 112, 144, 288, 32,  64, 64, **kwargs)
    self.inception4e = inception(528, 256, 160, 320, 32, 128, 128, **kwargs)

    self.inception5a = inception(832, 256, 160, 320, 32, 128, 128, **kwargs)
    self.inception5b = inception(832, 384, 192, 384, 48, 128, 128, **kwargs)
    
    if self.aux:
      self.aux1 = AuxClassifier(512, nc)
      self.aux2 = AuxClassifier(528, nc)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1024, nc))
    
    self.maxpool = nn.MaxPool2d(3, 2, ceil_mode=True)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)

    x = self.inception3a(x)
    x = self.inception3b(x)
    
    x = self.maxpool(x)

    x = self.inception4a(x)
    x = self.inception4b(x)
    x = self.inception4c(x)
    x1 = self.inception4d(x)
    x2 = self.inception4e(x1)
    
    # TODO Add the auxiliary classifiers in the forward pass

    x = self.maxpool(x2)

    x = self.inception5a(x)
    x = self.inception5b(x)
    
    x = self.avgpool(x)
    x = Flatten()(x)
    x = self.classifier(x)
    return x


def googlenet(pretrained=True, **kwargs):
  """
  GoogLeNet Model with weights as given by the officials who 
  trained it on TensorFlow.
  """
  kwargs['aux'] = False if 'aux' not in kwargs else kwargs['aux']
  kwargs['replace5x5with3x3'] = True if 'replace5x5with3x3' not in kwargs \
                                else kwargs['replace5x5with3x3']

  return get_net(GoogLeNet, pretrained=pretrained, fname='googlenet', 
                 kwargs_net=kwargs, attr='classifier', inn=1024,
                 pretrain_url=urls.googlenet_url)


def googlenet_paper(pretrained=False, **kwargs):
  """
  GoogLeNet Model as given in the official Paper.
  """
  if pretrained: raise Exception('No pretrained model available!')
  kwargs['aux'] = True if 'aux' not in kwargs else kwargs['aux']
  kwargs['replace5x5with3x3'] = False if 'replace5x5with3x3' not in kwargs \
                                else kwargs['replace5x5with3x3']

  return get_net(GoogLeNet, pretrained=pretrained, pretrain_url=None, 
                 fname='googlenet_paper', kwargs_net=kwargs, attr='classifier',
                 inn=1024)
