"""
Image Transformation Network from Justin Johnson et al. 2016. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import CenterCrop

from scratchai.nets.clf.resnet import conv as _conv
from scratchai.nets.seg.enet import uconv as _uconv

def conv(ic:int, oc:int, k:int=3, s:int=2, act:bool=True, 
         norm:bool=True) -> list:
  layers = [nn.ReflectionPad2d(k//2), nn.Conv2d(ic, oc, k, s, bias=True)]
  if norm: layers += [nn.InstanceNorm2d(oc, affine=True)]
  if act: layers += [nn.ReLU(inplace=True)]
  return layers


class uconv(nn.Module):
  """
  The Upsampling Module.

  This module doesn't uses ConvTranspose2d but instead uses nearest mode
  to upsample the image and then passes it through a Conv Layer as it is 
  shown to work better than ConvTranspose2d.
  Link: http://distill.pub/2016/deconv-checkerboard/

  Arguments
  ---------
  ic : int
       # of input channels
  oc : int
       # of output channels
  k : int, tuple
      Kernel_size
  s : int, tuple
      stride
  p : int, tuple
      Padding.
  up : int
       The upsampling factor
  """
  def __init__(self, ic:int, oc:int, k:int=3, s:int=1, act:nn.Module=nn.ReLU, 
               norm:nn.Module=nn.InstanceNorm2d, up:int=2):
    super().__init__()
    self.conv = nn.Sequential(*conv(ic, oc, k, s, act, norm))
    self.up = up
  def forward(self, x): return self.conv(F.interpolate(x, mode='nearest',
                                         scale_factor=self.up))


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
    self.main = nn.Sequential(*conv(ic, oc, s=1), *conv(oc, oc, s=1, act=None))
  def forward(self, x): return self.main(x) +  x


class ITN_ST(nn.Module):
  """
  This module implements the Image Transformation Network proposed in the
  Perceptual Losses paper by Justin et al. 2016 with tweaks.

  Paper Link: 
  https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf

  Notes
  -----
  This module is the recommended implementation of the model. Based on
  researches done after the release of the paper.
  """

  def __init__(self):
    super().__init__()
    layers = [*conv(3, 32, 9, 1), *conv(32, 64), *conv(64, 128)]
    layers += [resblock(128, 128, norm=nn.InstanceNorm2d) for _ in range(5)]
    layers += [uconv(128, 64), uconv(64, 32), uconv(32, 3, 9, up=1, norm=None, act=None)]
    self.net = nn.Sequential(*layers)

  def forward(self, x): return self.net(x)    

# Below is the original implementation of the Transformation Network
# as proposed by Justin et al. 2016. Research after the release of that paper
# shows the above architecture to work best. And the above architecture is made
# available by default in scratchai.  The below network can be called using 
# net = scratchai.nets.ITN_ST_()

class _resblock(nn.Module):
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
    self.main = nn.Sequential(*_conv(ic, oc, 3, 1, 0, norm=norm), 
                              *_conv(oc, oc, 3, 1, 0, act=None, norm=norm))
    self.act = nn.ReLU(inplace=True)

  def forward(self, x):
    return self.act(self.main(x) +  x[:, :, 2:-2, 2:-2])

class ITN_ST_(nn.Module):
  """
  This module implements the Image Transformation Network as proposed in the
  Perceptual Losses paper by Justin et al. 2016

  Paper Link: 
  https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf

  Notes
  -----
  This is the original implementation of the transformation network.
  """

  def __init__(self):
    super().__init__()

    layers = [nn.ReflectionPad2d(40), *_conv(3, 32, 9, 1, 4), 
              *_conv(32, 64, 3, 2, 1), *_conv(64, 128, 3, 2, 1)]
    layers += [_resblock(128, 128, norm=nn.InstanceNorm2d) for _ in range(5)]
    layers += [*_uconv(128, 64, 4, 2, 1), *_uconv(64, 32, 4, 2, 1), 
               *_uconv(32, 3, 9, 1, 4)]

    self.net = nn.Sequential(*layers)

  def forward(self, x): return self.net(x)
