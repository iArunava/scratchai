"""
Deep Convolutional GAN
Paper: https://arxiv.org/pdf/1511.06434.pdf
"""

import torch
import torch.nn as nn

from torch.nn import functional as F


__all__ = ['G']#, 'D']


def conv(ic:int, oc:int, k:int, s:int, p:int):
  layers = [nn.ConvTranspose2d(ic, oc, k, s, p, bias=False), nn.BatchNorm2d(oc)]
  return layers
            
class G(nn.Module):
  """
  Generator for the DCGAN.

  Arguments
  ---------

  """
  def __init__(self, sc:int=512, zc:int=100):
    super().__init__()
    layers_dict = nn.ModuleDict()
    prefix = 'conv'
    layers_dict[prefix + '0'] = nn.Sequential(*conv(zc, sc, 4, 1, 0))
    for ii in range(3):
      oc = sc // (2<<ii)
      ic = oc * 2
      layers_dict[prefix + str(ii+1)] = nn.Sequential(*conv(ic, oc, 4, 2, 1))

    self.layers_dict = layers_dict
    self.fconv = nn.Sequential(*conv(oc, 3, 4, 2, 1))

  def forward(self, x):
    for name, layer in self.layers_dict.items():
      x = F.relu(layer(x), inplace=True)
    x = F.tanh(self.fconv(x))
    return x
