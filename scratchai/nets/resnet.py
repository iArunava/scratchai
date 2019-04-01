import torch
import torch.nn as nn

from .blocks.resblock import *
from .blocks.bnconv import bnconv

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet50', 'resnet101', 'resnet152', 'resnext18', 'resnext34', 'resnext50', 'resnext101', 'resnext152']

class Resnet(nn.Module):
  
  def __init__(self, nc:int, layers:list, lconv:int=2, expansion:int=1, 
               block:nn.Module=Resblock, s1_channels:int=64, conv_first=True,
               inplace=True):
    '''
    The class that defines the ResNet module
    
    Arguments:
    - nc :: # of classes
    - s1_channels : # of channels for the output of the first stage
    - layers
    - lconv : # of conv layers in each Residual Block
    '''
    super(Resnet, self).__init__()
    
    layers = [bnconv(in_channels=3,
                out_channels=s1_channels,
                kernel_size=7,
                padding=3,
                stride=2,
                conv_first=conv_first),
    
              nn.MaxPool2d(kernel_size=3,
                                stride=2,
                                padding=1),
    
              res_stage(block, ic=s1_channels,
                              oc=s1_channels*expansion,
                              num_layers=layers[0],
                              stride=1, lconv=lconv,
                              conv_first=conv_first, inplace=inplace),
    
              res_stage(block, ic=s1_channels*expansion,
                              oc=s1_channels*expansion*2,
                              num_layers=layers[1], 
                              conv_first=conv_first, lconv=lconv,
                              inplace=inplace),
    
              res_stage(block, ic=s1_channels*expansion*2,
                              oc=s1_channels*expansion*4,
                              num_layers=layers[2], 
                              conv_first=conv_first, lconv=lconv,
                              inplace=inplace),
    
              res_stage(block, ic=s1_channels*expansion*4,
                              oc=s1_channels*expansion*8,
                              num_layers=layers[3], 
                              conv_first=conv_first, lconv=lconv,
                              inplace=inplace),
            ]
    
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
