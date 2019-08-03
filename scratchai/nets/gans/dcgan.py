"""
Deep Convolutional GAN
Paper: https://arxiv.org/pdf/1511.06434.pdf
"""

import torch
import torch.nn as nn

from torch.nn import functional as F


__all__ = ['G', 'D']


def convt(ic:int, oc:int, k:int, s:int, p:int):
  layers = [nn.ConvTranspose2d(ic, oc, k, s, p, bias=False), nn.BatchNorm2d(oc)]
  return layers


def conv(ic:int, oc:int, k:int, s:int, p:int):
  layers = [nn.Conv2d(ic, oc, k, s, p, bias=False), nn.BatchNorm2d(oc)]
  return layers


def create_layers(ic, sc, conv0_s, conv0_p, fconv_s, fconv_p, conv_func, foc, 
                  hid_layers=3):
  layers_dict = nn.ModuleDict()
  prefix = 'conv'
  layers_dict[prefix + '0'] = nn.Sequential(*conv_func(ic, sc, 4, conv0_s, conv0_p))
  for ii in range(hid_layers):
    oc = sc // (2<<ii)
    ic = oc * 2
    layers_dict[prefix + str(ii+1)] = nn.Sequential(*conv_func(ic, oc, 4, 2, 1))
  fconv = nn.Sequential(*conv_func(oc, foc, 4, fconv_s, fconv_p))
  return layers_dict, fconv

 
class G(nn.Module):
  """
  Generator for the DCGAN.

  Arguments
  ---------

  """
  def __init__(self, sc:int=512, zc:int=100, **kwargs):
    super().__init__()
    self.layers_dict, self.fconv = create_layers(zc, sc, 1, 0, 2, 1, convt, 3, **kwargs)

  def forward(self, x):
    for name, layer in self.layers_dict.items():
      x = F.relu(layer(x), inplace=True)
    x = F.tanh(self.fconv(x))
    return x


class D(nn.Module):
  """
  Discriminator for the DCGAN.

  Arguments
  ---------

  """
  def __init__(self, sc:int=64, expand:int=2, **kwargs):
    super().__init__()
    self.layers_dict, self.fconv = create_layers(3, sc, 2, 1, 1, 0, conv, 1, **kwargs)

  def forward(self, x):
    for name, layer in self.layers_dict.items():
      x = F.leaky_relu(layer(x), negative_slope=0.2, inplace=True)
    x = F.sigmoid(self.fconv(x))
    x = x.view(-1, 1)
    return x
