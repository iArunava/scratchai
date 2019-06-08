"""
FCN - Fully Convolutional Neural Networks
Paper: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
"""

import torch
import torch.nn as nn

from scratchai import nets


__all__ = ['fcn_alexnet', 'FCNHead']


class FCNHead(nn.Module):
  def __init__(self, ic:int, oc:int=21, compress:int=4):
    super().__init__()
    inter_channels = ic // compress

    self.net = nn.Sequential(
      nn.Conv2d(ic, inter_channels, 3, 1, 1, bias=False),
      nn.BatchNorm2d(inter_channels),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.1),
      nn.Conv2d(inter_channels, oc, 1, 1, 0)
    )

  def forward(self, x): return self.net(x)


class FCN(nn.Module):
  """
  Implementation of the FCN model.
  """
  def __init__(self, head_ic:int, nc=21, backbone=None):
    super().__init__()
    self.net = nn.Sequential(backbone, FCNHead(ic=head_ic))

  def forward(self, x):
    return x


def fcn_alexnet():
  net = nets.alexnet()
  backbone = net.net[:13]
  return FCN(head_ic=256, backbone=backbone)
