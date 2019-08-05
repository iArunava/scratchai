"""
Deep Convolutional GAN
Paper: https://arxiv.org/pdf/1511.06434.pdf
"""

import torch
import torch.nn as nn

from torch.nn import functional as F

from scratchai.init import dcgan_init


__all__ = ['G', 'D', 'get_dcgan', 'get_mnist_dcgan']


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
  def __init__(self, sc:int=512, zc:int=100, oc:int=3, **kwargs):
    super().__init__()
    self.layers_dict, self.fconv = create_layers(zc, sc, 1, 0, 2, 1, convt, oc, **kwargs)
    self.apply(dcgan_init)

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
  def __init__(self, sc:int=64, expand:int=2, ic:int=3, **kwargs):
    super().__init__()
    self.layers_dict, self.fconv = create_layers(ic, sc, 2, 1, 1, 0, conv, 1, **kwargs)
    self.apply(dcgan_init)

  def forward(self, x):
    for name, layer in self.layers_dict.items():
      x = F.leaky_relu(layer(x), negative_slope=0.2, inplace=True)
    x = F.sigmoid(self.fconv(x))
    x = x.view(-1, 1)
    return x


# =============================================================================
# QuickLoaders
# =============================================================================
def get_mnist_dcgan():
  """
  This function returns the Generator and Discriminator as needed by MNIST 
  dataset.
  """
  return G(hid_layers=2, oc=1), D(hid_layers=2, ic=1)


def get_dcgan(scale='normal'):
  """
  This function returns the Generator and Discriminator as described in DCGAN
  """
  if scale == 'small':
    return G(sc=4, hid_layers=1), D(sc=4, hid_layers=1, ic=1)
  elif scale == 'normal':
    return G(), D()
  else:
    raise NotImplementedError
  
