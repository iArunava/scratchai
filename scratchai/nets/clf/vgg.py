"""
VGG Model
"""

import torch
import torch.nn as nn
import numpy as np

from scratchai.nets.utils import get_net
from scratchai.nets.common import Flatten
from scratchai.pretrained import urls

__all__ = ['VGG', 'vgg11', 'vgg_dilated', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 
           'vgg16_bn', 'vgg19', 'vgg19_bn', 'vgg_block']


def vgg_block(ic:int, oc:int, k:int=3, s:int=1, d:int=1, norm:bool=True):
  # NOTE This overwrites the padding
  p = d
  layers = [nn.Conv2d(ic, oc, k, s, p, dilation=d, bias=True)]
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
  def __init__(self, nc=1000, lconf:list=[1, 1, 2, 2, 2], ic:int=3, 
               norm:bool=True, dilation:int=1, downx:int=32, dilate_last:int=0):
    super().__init__()
    
    ic = ic; oc = 64
    pad = 1
    features = []
    blocks = len(lconf)
    pool_stride = 2
    curr_dilation = 1
    ps1_last = blocks - np.log2(downx)
    for ii, l in enumerate(lconf):
      if ii >= (blocks-ps1_last): pool_stride = 1
      if ii >= (blocks-dilate_last): curr_dilation = dilation

      features += vgg_block(ic, oc, d=curr_dilation, norm=norm)
      ic = oc
      for _ in range(l-1):
        features += vgg_block(ic, oc, d=curr_dilation, norm=norm)
      oc *= 2 if oc*2 <= 512 else 1
      features += [nn.MaxPool2d(3, pool_stride, padding=1)]
    
    self.features = nn.Sequential(*features)
    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    self.classifier = nn.Sequential(Flatten(), *linear(512 * 7 * 7, 4096), 
                                    *linear(4096, 4096), nn.Linear(4096, nc))

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = self.classifier(x)
    return x

  
# =======================================================================
# VGG Dilated
# =======================================================================
def vgg_dilated(vgg='vgg11', pretrained=True, **kwargs):
  kwargs['dilation'] = 2
  kwargs['dilate_last'] = 1
  kwargs['downx'] = 8
  return globals()[vgg](pretrained=pretrained, **kwargs)


# =======================================================================
# VGG A 
# =======================================================================
def vgg11(pretrained=True, **kwargs):
  kwargs['lconf'] = [1, 1, 2, 2, 2]
  kwargs['norm']  = False
  return get_net(VGG, pretrained=pretrained, pretrain_url=urls.vgg11, 
                 fname='vgg11', kwargs_net=kwargs, attr='classifier',
                 inn=25088)

def vgg11_bn(pretrained=True, **kwargs):
  kwargs['lconf'] = [1, 1, 2, 2, 2]
  return get_net(VGG, pretrained=pretrained, pretrain_url=urls.vgg11_bn, 
                 fname='vgg11_bn', kwargs_net=kwargs, attr='classifier',
                 inn=25088)

# =======================================================================
# VGG B
# =======================================================================
def vgg13(pretrained=True, **kwargs):
  kwargs['lconf'] = [2, 2, 2, 2, 2]
  kwargs['norm']  = False
  return get_net(VGG, pretrained=pretrained, pretrain_url=urls.vgg13, 
                 fname='vgg13', kwargs_net=kwargs, attr='classifier',
                 inn=25088)

def vgg13_bn(pretrained=False, **kwargs):
  if pretrained: raise Exception('No pretrained models avaialble!')
  kwargs['lconf'] = [2, 2, 2, 2, 2]
  return get_net(VGG, pretrained=pretrained, pretrain_url=urls.vgg13, 
                 fname='vgg13', kwargs_net=kwargs, attr='classifier',
                 inn=25088)

# =======================================================================
# VGG D
# =======================================================================
def vgg16(pretrained=True, **kwargs):
  kwargs['lconf'] = [2, 2, 3, 3, 3]
  kwargs['norm']  = False
  return get_net(VGG, pretrained=pretrained, pretrain_url=urls.vgg16, 
                 fname='vgg16', kwargs_net=kwargs, attr='classifier',
                 inn=25088)
 
def vgg16_bn(pretrained=True, **kwargs):
  kwargs['lconf'] = [2, 2, 3, 3, 3]
  return get_net(VGG, pretrained=pretrained, pretrain_url=urls.vgg16_bn,
                 fname='vgg16_bn', kwargs_net=kwargs, attr='classifier',
                 inn=25088)

# =======================================================================
# VGG E
# =======================================================================
def vgg19(pretrained=True, **kwargs):
  kwargs['lconf'] = [2, 2, 4, 4, 4]
  kwargs['norm']  = False
  return get_net(VGG, pretrained=pretrained, pretrain_url=urls.vgg19,
                 fname='vgg19', kwargs_net=kwargs, attr='classifier',
                 inn=25088)
  
def vgg19_bn(pretrained=False, **kwargs):
  if pretrained: raise Exception('No pretrained models avaialble!')
  kwargs['lconf'] = [2, 2, 4, 4, 4]
  return get_net(VGG, pretrained=pretrained, pretrain_url=None,
                 fname='vgg19_bn', kwargs_net=kwargs, attr='classifier',
                 inn=25088)



