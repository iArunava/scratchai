"""
PSPNet.
Paper: https://arxiv.org/pdf/1612.01105.pdf
"""

import torch
import torch.nn as nn

from torch.nn import functional as F
from collections import OrderedDict

from scratchai import nets
from scratchai.nets.common import InterLayer


__all__ = ['PyramidPooling', 'pspnet_alexnet', 'pspnet_resnet101']



class PyramidPooling(nn.Module):
  """
  The Pyramid Pooling Module as proposed by the authors of PSPNet.

  Paper: https://arxiv.org/pdf/1612.01105.pdf

  Arguments
  ---------

  """
  def __init__(self, ic:int, pyramids:tuple=(1, 3, 4, 6)):
    super().__init__()
    self.ic = ic
    self.oc = ic // len(pyramids)
    assert(self.oc > 0)
    
    self.pyramids = nn.ModuleList()
    for p in pyramids:
      self.pyramids.append(self._make_stage(p))

  def _make_stage(self, size):
    pool = nn.AdaptiveAvgPool2d(output_size=(size, size))
    conv = nn.Conv2d(self.ic, self.oc, 1, 1, 0)
    return nn.Sequential(pool, conv)

  def forward(self, feats):
    oshape = feats.shape[-2:]
    feats = [F.interpolate(layer(feats), size=oshape, mode='bilinear', \
             align_corners=False) for layer in self.pyramids] + [feats]
    out = torch.cat(feats, dim=1)
    return F.relu(out, inplace=True)


class Upsample(nn.Module):
  def __init__(self, ic, oc):
    super().__init__()
    self.net = nn.Sequential(nn.Conv2d(ic, oc, 1, 1, 0),
                             nn.BatchNorm2d(oc), nn.PReLU())
    
  def forward(self, x):
    x = self.net(x)
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    return x


class PSPNet(nn.Module):
  def __init__(self, backbone:nn.Module, nc:int=21, ic:int=3, head_ic:int=1024):
    super().__init__()
    self.backbone = backbone
    self.aux_clf = None
    head_ic = head_ic
    self.head = PyramidPooling(head_ic)

    layers_dict = self.backbone.get_oc_for(self.backbone.return_layers)
    self.drop1 = nn.Dropout2d(0.3, inplace=True)
    self.drop2 = nn.Dropout2d(0.15, inplace=True)
    
    up_ic = 2 * head_ic; up_oc = (2 * head_ic) // 4
    self.uplist = nn.ModuleList()
    for ii in range(3):
      self.uplist.append(Upsample(up_ic, up_oc))
      # Validate this Step
      if ii != 2:
        up_ic //= 4
        up_oc //= 4

    self.fconv = nn.Sequential(nn.Conv2d(up_oc, nc, 3, 1, 1), nn.LogSoftmax())


  def forward(self, x):
    x_shape = x.shape[-2:]
    
    out = OrderedDict()
    features_out = self.backbone(x)
    print (features_out['out'].shape)

    if self.aux_clf is not None and 'aux' in features_out:
      out['aux'] = self.aux_clf(features_out['aux'])
    
    out['out'] = self.head(features_out['out'])

    out = self.head(features_out['out'])

    for layer in self.uplist:
      out = self.drop2(layer(out))
    
    out = self.fconv(out)
    return out


def get_pspnet(nc, aux, features, return_layers:dict, **kwargs):
  """
  Helper Function to Get a FCN.

  Arguments
  ---------
  nc : int
       The number of classes.

  features: nn.Module
            The backbone.

  return_layers : dict
               Dict containing all the other layer names along with 
               what name they should have in the output dictionary.

  head_ic : int
            The number of in_channels to the FCN Head.
  """
  backbone = InterLayer(features, return_layers)
  aux_classifier = PyramidPooling(head_ic, oc=nc) if aux else None
  return PSPNet(backbone=backbone, nc=nc, **kwargs)


def pspnet_alexnet(nc=21, aux:bool=False, **kwargs):
  return_layers = {'12': 'out'}
  if aux: return_layers ['9'] = 'aux'
  return get_pspnet(nc, aux, nets.clf.alexnet().features, return_layers, head_ic=256, **kwargs)


def pspnet_resnet101(nc=21, aux:bool=False, **kwargs):
  return_layers = {'36': 'out'}
  if aux: return_layers ['22'] = 'aux'
  return get_pspnet(nc, aux, nets.clf.resnet101().features, return_layers, head_ic=2048, **kwargs)
