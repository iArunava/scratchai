"""
The Resnet
"""

import torch
import torch.nn as nn

from scratchai.nets.blocks.resblock import Resnext
from scratchai.nets.blocks.bnconv import bnconv


__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet50', 'resnet101', 'resnet152', 'resnext18', 'resnext34', 'resnext50', 'resnext101', 'resnext152']


def conv(ic:int, oc:int, ks:int=3, s:int=1, p:int=1, norm:nn.Module=nn.BatchNorm2d, act:bool=True):
  layers = [nn.Conv2d(ic, oc, kernel_size=ks, stride=s, padding=p, bias=not norm)]
  if norm: layers += [norm(oc)]
  if act: layers += [nn.ReLU(inplace=True)]
  return layers


class resblock(nn.Module):
  """
  This module is the implementation of a resnet block.

  Arguments
  ---------
  ic : int
       # of input channels
  oc : int
       # of output channels
  ks : int
       kernel_size of the mid_conv
  s : int
      stride of the mid_conv
  p : int
      padding of the mid_conv
  norm : nn.Module
         The Normalization to be used.
  act : bool
        whether to use activation
  pratio : int
           the factor by which the # of in_channels is reduced
  layers : int
           IDEA layers is not yet implemented but the idea is that the
           main branch will contain conv layers equal to the number of layers.
           Helpful in research purposes.
  """

  def __init__(self, ic:int, oc:int=None, norm:nn.Module=nn.BatchNorm2d, 
        act:bool=True, dflag:bool=False, pratio:int=4, layers:int=2):
    super().__init__()
    self.dflag = dflag
    s = 2 if dflag else 1; oc = ic*2 if not oc else oc
    self.main = nn.Sequential(*conv(ic, oc), \
                              *conv(oc, oc, s=s, act=None))
    self.act = nn.ReLU(inplace=True)
    self.side = conv(ic, oc, s=s, act=None) if dflag else None

  def forward(self, x): 
    return self.act(self.main(x) + (self.side(x) if self.dflag else x))


def res_stage(block:nn.Module, ic:int, oc:int, num_layers:int):

  """

  Arguments
  ---------
  
  block : nn.Module
          the block type to be stacked one upon another
  ic : int
       # of input channels
  oc : int
       # of output channels
  num_layers - int
               # of blocks to be stacked
  """

  layers = []
  layers += nn.ModuleList([block(ic=ic, oc=oc, dflag=True)])

  layers.extend([block(ic=oc, oc=oc) for i in range (num_layers-1)])
  return nn.Sequential(*layers)


class Resnet(nn.Module):
  """
  The class that defines the ResNet module.

  Arguments:
  nc : int
       # of classes
  oc1 : int
                # of channels for the output of the first stage
  layers : TOFILL
  lconv : int
          # of conv layers in each Residual Block
  """

  def __init__(self, nc:int, layers:list, lconv:int=2, ex:int=1, 
       block:nn.Module=resblock, oc1:int=64, conv_first=True,
       inplace=True):
    super(Resnet, self).__init__()

    layers = [*conv(3, oc1, 7, 2, 3),
              nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
              res_stage(block, oc1, oc1*ex, layers[0]),
              res_stage(block, oc1*ex, oc1*ex*2, layers[1]),
              res_stage(block, oc1*ex*2, oc1*ex*4, layers[2]),
              res_stage(block, oc1*ex*4, oc1*ex*8, layers[3])]
    self.net = nn.Sequential(*layers)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, nc)

  def forward(self, x):
    bs = x.size(0)
    x = self.net(x)
    x = self.avgpool(x) if self.apool else x
    x = x.view(bs, -1)
    x = self.fc(x)
    return x


def resnet18(nc, **kwargs):
  kwargs['nc'] = nc
  kwargs['layers'] = [2, 2, 2, 2]
  kwargs['lconv'] = 2
  return Resnet(**kwargs)

def resnet34(nc, **kwargs):
  kwargs['nc'] = nc
  kwargs['layers'] = [3, 4, 6, 3]
  kwargs['lconv'] = 2
  return Resnet(**kwargs)

def resnet50(nc, **kwargs):
  kwargs['nc'] = nc
  kwargs['layers'] = [3, 4, 6, 3]
  kwargs['lconv'] = 3
  return Resnet(**kwargs)

def resnet101(nc, **kwargs):
  kwargs['nc'] = nc
  kwargs['layers'] = [3, 4, 23, 3]
  kwargs['lconv'] = 3
  return Resnet(**kwargs)

def resnet152(nc, **kwargs):
  kwargs['nc'] = nc
  kwargs['layers'] = [3, 8, 56, 3]
  kwargs['lconv'] = 3
  return Resnet(**kwargs)

def resnext18(nc, **kwargs):
  kwargs['nc'] = nc
  kwargs['layers'] = [2, 2, 2, 2]
  kwargs['lconv'] = 2
  kwargs['block'] = Resnext
  return Resnet(**kwargs)

def resnext34(nc, **kwargs):
  kwargs['nc'] = nc
  kwargs['layers'] = [3, 4, 6, 3]
  kwargs['lconv'] = 2
  kwargs['block'] = Resnext
  return Resnet(**kwargs)

def resnext50(nc, **kwargs):
  kwargs['nc'] = nc
  kwargs['layers'] = [3, 4, 6, 3]
  kwargs['lconv'] = 3
  kwargs['block'] = Resnext
  return Resnet(**kwargs)

def resnext101(nc, **kwargs):
  kwargs['nc'] = nc
  kwargs['layers'] = [3, 4, 23, 3]
  kwargs['lconv'] = 3
  kwargs['block'] = Resnext
  return Resnet(**kwargs)

def resnext152(nc, **kwargs):
  kwargs['nc'] = nc
  kwargs['layers'] = [3, 8, 56, 3]
  kwargs['lconv'] = 3
  kwargs['block'] = ResneXt
  return Resnet(**kwargs)
