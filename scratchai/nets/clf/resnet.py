"""
The Resnet
"""

import torch
import torch.nn as nn
import scratchai.pretrained.urls as urls

from scratchai.utils import load_from_pth, load_pretrained
from scratchai.nets.blocks.resblock import Resnext


__all__ = ['resnet18_mnist', 'resnet18', 'resnet34', 'resnet50', 'resnet50', 
           'resnet101', 'resnet152', 'resnext18', 'resnext34', 'resnext50', 
           'resnext101', 'resnext152']


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
        act:bool=True, dflag:bool=False, pratio:int=4, layers:int=2, 
        btype:str='basic', fdown:bool=False):
    super().__init__()
    assert btype in ['basic', 'bottleneck']
    self.dflag = dflag
    s = 2 if dflag else 1; oc = ic*2 if not oc else oc
    
    if btype == 'basic':
        self.main = nn.Sequential(*conv(ic, oc, s=s), *conv(oc, oc, act=None))
    elif btype == 'bottleneck':
        interc = oc // 4
        self.main = nn.Sequential(*conv(ic, interc, 1, 1, 0), \
                                  *conv(interc, interc, s=s),
                                  *conv(interc, oc, 1, 1, 0, act=None),)
    self.act = nn.ReLU(inplace=True)
    
    # HACK fdown is introduced just because the 1st layer of a resnet >= 50
    # needs a side branch with a 1x1 convolution. Find a better way if possible.
    if dflag or fdown:
      self.side = nn.Sequential(*conv(ic, oc, 1, s, 0, act=None))
    else:
      self.side = None

  def forward(self, x):
    return self.act(self.main(x) + (self.side(x) if self.side else x))


def res_stage(block:nn.Module, ic:int, oc:int, num_layers:int, dflag:bool=True,
              btype:str='basic', fdown:bool=False):

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
  dflag : bool
          Whether the first resblock needs to perform downsampling.
          Defaults to True.
  btype : str, should be one of ['basic'. 'bottleneck']
          The type of resblock to be used. Defaults to 'basic'
  fdown : bool
          If true the side branch *must* contain a conv block,
          whether it performs downsampling or not. Defaults to False.

  Returns
  -------
  layers : list
           A list containing all the nn.Module that is required for this layer.
  """
  
  layers = [block(ic, oc, dflag=dflag, btype=btype, fdown=fdown)]
  layers += [block(oc, oc, btype=btype) for i in range (num_layers-1)]
  return layers


class Resnet(nn.Module):
  """
  The class that defines the ResNet module.

  Arguments:
  nc : int
       # of classes
  ic : int
       # of input channels to the network
  oc1 : int
        # of channels for the output of the first stage
  layers : TOFILL
  lconv : int
          # of conv layers in each Residual Block
  """

  def __init__(self, layers:list, nc:int=1000, lconv:int=2, ex:int=1, 
       block:nn.Module=resblock, ic:int=3, oc1:int=64, **kwargs):
    super().__init__()

    layers = [*conv(ic, oc1, 7, 2, 3),
              nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
              *res_stage(block, oc1, oc1*ex, layers[0], dflag=False, **kwargs),
              *res_stage(block, oc1*ex, oc1*ex*2, layers[1], **kwargs),
              *res_stage(block, oc1*ex*2, oc1*ex*4, layers[2], **kwargs),
              *res_stage(block, oc1*ex*4, oc1*ex*8, layers[3], **kwargs),
              nn.AdaptiveAvgPool2d((1, 1))]
    self.net = nn.Sequential(*layers)
    
    self.fc = nn.Linear(512*ex, nc)

  def forward(self, x):
    bs = x.size(0)
    x = self.net(x)
    x = x.view(bs, -1)
    x = self.fc(x)
    return x


def resnet18_mnist(pretrained=False, **kwargs):
  kwargs['layers'] = [2, 2, 2, 2]
  kwargs['ic'] = 1
  net = Resnet(**kwargs)
  '''
  if pretrained:
    # TODO check inspect module and change the fname
    net.load_state_dict(load_from_pth(urls.resnet18_url, 'resnet18'))
  '''
  return net
  
def resnet18(pretrained=True, **kwargs):
  kwargs['layers'] = [2, 2, 2, 2]
  cust_nc = None
  if pretrained and 'nc' in kwargs: cust_nc = kwargs['nc']; kwargs['nc'] = 1000
  net = Resnet(**kwargs)
  if pretrained:
    # TODO check inspect module and change the fname
    return load_pretrained(net, urls.resnet18_url, 'resnet18', nc=cust_nc)
  return net

def resnet34(pretrained=True, **kwargs):
  kwargs['layers'] = [3, 4, 6, 3]
  cust_nc = None
  if pretrained and 'nc' in kwargs: cust_nc = kwargs['nc']; kwargs['nc'] = 1000
  net = Resnet(**kwargs)
  if pretrained:
    return load_pretrained(net, urls.resnet34_url, 'resnet34', nc=cust_nc)
  return net

def resnet50(pretrained=True, **kwargs):
  kwargs['layers'] = [3, 4, 6, 3]
  kwargs['btype'] = 'bottleneck'
  kwargs['ex'] = 4; kwargs['fdown'] = True
  cust_nc = None
  if pretrained and 'nc' in kwargs: cust_nc = kwargs['nc']; kwargs['nc'] = 1000
  net = Resnet(**kwargs)
  if pretrained:
    return load_pretrained(net, urls.resnet50_url, 'resnet50', nc=cust_nc, 
                           inn=512*kwargs['ex'])
  return net

def resnet101(pretrained=True, **kwargs):
  kwargs['layers'] = [3, 4, 23, 3]
  kwargs['btype'] = 'bottleneck'
  kwargs['ex'] = 4; kwargs['fdown'] = True
  cust_nc = None
  if pretrained and 'nc' in kwargs: cust_nc = kwargs['nc']; kwargs['nc'] = 1000
  net = Resnet(**kwargs)
  if pretrained:
    return load_pretrained(net, urls.resnet101_url, 'resnet101', nc=cust_nc,
                           inn=512*kwargs['ex'])
  return net

def resnet152(pretrained=True, **kwargs):
  kwargs['layers'] = [3, 8, 36, 3]
  kwargs['btype'] = 'bottleneck'
  kwargs['ex'] = 4; kwargs['fdown'] = True
  cust_nc = None
  if pretrained and 'nc' in kwargs: cust_nc = kwargs['nc']; kwargs['nc'] = 1000
  net = Resnet(**kwargs)
  if pretrained:
    return load_pretrained(net, urls.resnet152_url, 'resnet152', nc=cust_nc,
                           inn=512*kwargs['ex'])
  return net

# FIXME The resnet blocks work okay but resnext needs a check
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
