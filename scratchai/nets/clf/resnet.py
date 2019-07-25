"""
The Resnet
"""

import torch
import torch.nn as nn
import scratchai.pretrained.urls as urls
import numpy as np

from scratchai.utils import load_from_pth, load_pretrained


__all__ = ['resnet18_mnist', 'resnet18', 'resnet34', 'resnet50', 'resnet50', 
           'resnet101', 'resnet152', 'resnet_dilated']


def conv(ic:int, oc:int, ks:int=3, s:int=1, p:int=1, d:int=1, norm:nn.Module=nn.BatchNorm2d, act:bool=True):
  p = d if d > 1 else p
  layers = [nn.Conv2d(ic, oc, ks, s, p, dilation=d, bias=not norm)]
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
        btype:str='basic', fdown:bool=False, dilation:int=1):
    super().__init__()
    assert btype in ['basic', 'bottleneck']
    self.dflag = dflag
    s = 2 if dflag else 1; oc = ic*2 if not oc else oc
    
    if btype == 'basic':
        self.main = nn.Sequential(*conv(ic, oc, s=s), *conv(oc, oc, act=None))
    elif btype == 'bottleneck':
        interc = oc // 4
        self.main = nn.Sequential(*conv(ic, interc, 1, 1, 0), \
                                  *conv(interc, interc, s=s, d=dilation),
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
              btype:str='basic', fdown:bool=False, dilation:int=1, 
              downx:int=32):

  """

  Arguments
  ---------
  
  block : nn.Module
          the block type to be stacked one upon another
  ic : int
       # of input channels
  oc : int
       # of output channels
  num_layers : int
               # of blocks to be stacked
  dflag : bool
          Whether the first resblock needs to perform downsampling.
          Defaults to True.
  btype : str, should be one of ['basic'. 'bottleneck']
          The type of resblock to be used. Defaults to 'basic'
  fdown : bool
          If true the side branch *must* contain a conv block,
          whether it performs downsampling or not. Defaults to False.
  
  dilaltion : int
              The dilation to be applied to the the first conv of the branch_2b
              of each resblock.

  downx : int
          How much times the input size is to be reduced.
          So, if it is 8, and the input has a height and width of 224 then the 
          output is supposed to be 224/8

  Returns
  -------
  layers : list
           A list containing all the nn.Module that is required for this layer.
  """
  layers = [block(ic, oc, dflag=dflag, btype=btype, fdown=fdown, \
                  dilation=dilation)]
  layers += [block(oc, oc, btype=btype, dilation=dilation) \
             for i in range (num_layers-1)]
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

  def __init__(self, layers:list, nc:int=1000, lconv:int=2, ex:int=0,
       block:nn.Module=resblock, ic:int=3, oc1:int=64, dilate_last:int=0,
       downx:int=32, dilation:int=1, increase_dilation_by_2:bool=True, **kwargs):
    super().__init__()
    
    features = [*conv(ic, oc1, 7, 2, 3), nn.MaxPool2d(3, 2, 1)]
    blocks = len(layers)
    curr_dilation = 1
    ps1_last = blocks - np.log2(downx) + 1

    for ii, layer in enumerate(layers):
      dflag = False if ii == 0 else True
      if ii >= (blocks-ps1_last): dflag = False
      if ii >= (blocks-dilate_last):
        curr_dilation = dilation
        if increase_dilation_by_2:
          dilation *= 2
      
      features += res_stage(block, oc1*(1<<(ex*int(0<ii)))*(1<<max(0, ii-1)),  \
                            oc1*(1<<(ex*int(0<(ii+1))))*(1<<(ii)), \
                            layer, dflag=dflag, dilation=curr_dilation, **kwargs)


    self.features = nn.Sequential(*features)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512*(1<<ex), nc)
  
  def _make_stage(self, ii, ex, layers, dflag, **kwargs):
    # NOTE ex is expected to be a power of 2, of if the expansion is needed
    # to be 4, then ex should be 2
    return res_stage(block, oc1*(1<<(ex*int(0<ii)))*(1<<max(0, ii-1)),  \
                     oc1*(1<<(ex*int(0<(ii+1))))*(1<<(ii)), \
                     layers[ii], dflag=dflag, **kwargs)
  
  def forward(self, x):
    bs = x.size(0)
    x = self.features(x)
    x = self.avgpool(x)
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
  

def resnet_dilated(resnet='resnet50', pretrained=True, **kwargs):
  kwargs['dilation'] = 2
  kwargs['dilate_last'] = 2
  kwargs['increase_dilation_by_2'] = True
  kwargs['downx'] = 8
  return globals()[resnet](pretrained=pretrained, **kwargs)


# =============================================================================
# Resnet18
# =============================================================================
def resnet18(pretrained=True, **kwargs):
  kwargs['layers'] = [2, 2, 2, 2]
  cust_nc = None
  if pretrained and 'nc' in kwargs: cust_nc = kwargs['nc']; kwargs['nc'] = 1000
  net = Resnet(**kwargs)
  if pretrained:
    # TODO check inspect module and change the fname
    return load_pretrained(net, urls.resnet18_url, 'resnet18', nc=cust_nc)
  return net


# =============================================================================
# Resnet34
# =============================================================================
def resnet34(pretrained=True, **kwargs):
  kwargs['layers'] = [3, 4, 6, 3]
  cust_nc = None
  if pretrained and 'nc' in kwargs: cust_nc = kwargs['nc']; kwargs['nc'] = 1000
  net = Resnet(**kwargs)
  if pretrained:
    return load_pretrained(net, urls.resnet34_url, 'resnet34', nc=cust_nc)
  return net


# =============================================================================
# Resnet50
# =============================================================================
def resnet50(pretrained=True, **kwargs):
  kwargs['layers'] = [3, 4, 6, 3]
  kwargs['btype'] = 'bottleneck'
  kwargs['ex'] = 2; kwargs['fdown'] = True
  cust_nc = None
  if pretrained and 'nc' in kwargs: cust_nc = kwargs['nc']; kwargs['nc'] = 1000
  net = Resnet(**kwargs)
  if pretrained:
    return load_pretrained(net, urls.resnet50_url, 'resnet50', nc=cust_nc, 
                           inn=512*kwargs['ex'])
  return net


# =============================================================================
# Resnet101
# =============================================================================
def resnet101(pretrained=True, **kwargs):
  kwargs['layers'] = [3, 4, 23, 3]
  kwargs['btype'] = 'bottleneck'
  kwargs['ex'] = 2; kwargs['fdown'] = True
  cust_nc = None
  if pretrained and 'nc' in kwargs: cust_nc = kwargs['nc']; kwargs['nc'] = 1000
  net = Resnet(**kwargs)
  if pretrained:
    return load_pretrained(net, urls.resnet101_url, 'resnet101', nc=cust_nc,
                           inn=512*kwargs['ex'])
  return net


# =============================================================================
# Resnet152
# =============================================================================
def resnet152(pretrained=True, **kwargs):
  kwargs['layers'] = [3, 8, 36, 3]
  kwargs['btype'] = 'bottleneck'
  kwargs['ex'] = 2; kwargs['fdown'] = True
  cust_nc = None
  if pretrained and 'nc' in kwargs: cust_nc = kwargs['nc']; kwargs['nc'] = 1000
  net = Resnet(**kwargs)
  if pretrained:
    return load_pretrained(net, urls.resnet152_url, 'resnet152', nc=cust_nc,
                           inn=512*kwargs['ex'])
  return net

# FIXME The resnet blocks work okay but resnext needs a check
# Uncomment when fixed.
"""
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
"""
