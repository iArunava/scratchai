"""
FCN - Fully Convolutional Neural Networks
Paper: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from scratchai import nets
from scratchai.imgutils import center_crop
from scratchai.nets.common import InterLayer
from scratchai.utils import bilinear_kernel
from scratchai.init import zero_init


__all__ = ['FCNHead', 'fcn_alexnet', 'fcn_vgg', 'fcn_googlenet']


def conv(ic:int, oc:int, ks:int):
  layers = []
  layers.append(nn.Conv2d(ic, oc, ks, 1, 0))
  layers.append(nn.ReLU(inplace=True))
  layers.append(nn.Dropout2d(p=0.5))
  return layers


class FCNHead_Mod1(nn.Module):
  """
  The FCNHead Mod that was used in torchvision
  """
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

  def forward(self, x, shape):
    x = self.net(x)
    out = F.interpolate(x, size=shape, mode='bilinear', align_corners=False)
    return out


class FCNHead(nn.Module):
  """
  Original FCN Head as implemented by the FCN authors.
  
  Arguments
  ---------
  ic : int
       The number of in channels

  oc : int
       The number of out channels

  Note
  ----
  This head expects a padded input. So, say the input image should be padded 
  with 100.  So, that when the fixed bilinear Kernel (the ConvTranspose with 
  fixed bilinear weights) upsamples, it will be more than the size of the input
  and then the upsampled image is center cropped to get the required image size.

  Also, do note, that the required image size must be passed in while calling 
  the forward fuction.
  """
  def __init__(self, ic:int, oc:int=21):
    super().__init__()

    self.net = nn.Sequential(
                  *conv(ic, 4096, 6), *conv(4096, 4096, 1),
                   nn.Conv2d(4096, oc, 1, 1, 0),
                   nn.ConvTranspose2d(oc, oc, 63, stride=32, bias=False),
              )
    
    self.net.apply(zero_init)
    self.net[-1].weight.data.copy_(bilinear_kernel(oc, oc, 63))
    self.net[-1].weight.requires_grad_(False)

  def forward(self, x, shape): 
    x = self.net(x)
    # Cropping the image to the required size (as mentioned by shape)
    x = center_crop(x, shape)
    return x


class FCN(nn.Module):
  """
  Implementation of the FCN model.

  Arguemnts
  ---------
  head_ic        : int
                   The number of in_channels for the FCN Head

  backbone       : nn.Module
                   The Backbone of the FCN.

  aux_classifier : nn.Module
                   The Aux Classifier.

  pad_input      : bool
                   Whether to pad the incoming input or not.
  """
  def __init__(self, head_ic:int, nc=21, backbone=None, aux_classifier=None,
               pad_input:bool=False):
    super().__init__()
    self.pad_input = pad_input
    self.backbone = backbone
    self.fcn_head = FCNHead(ic=head_ic, oc=nc)
    self.aux_classifier = aux_classifier

  def forward(self, x):
    x_shape = x.shape[-2:]
    if self.pad_input:
      x = F.pad(x, (100, 100, 100, 100), mode='constant', value=0)
    
    out = OrderedDict()
    features_out = self.backbone(x)

    if self.aux_classifier is not None and 'aux' in features_out:
      out['aux'] = self.aux_classifier(features_out['aux'], x_shape)
    """
    if len(features_out) > 1:
      for key in features_out.keys():
        out['aux'] = self.aux_classifier(features_out['aux'], x_shape)
    """

    out['out'] = self.fcn_head(features_out['out'], x_shape)
    return out


# =============================================================================
# FCN-32s
# =============================================================================

# FCN32-Alexnet
def fcn_alexnet(nc=21, aux:bool=False):
  backbone = InterLayer(nets.alexnet().features, {'9': 'aux', '12': 'out'})
  aux_classifier = FCNHead(ic=256, oc=nc) if aux else None
  return FCN(head_ic=256, backbone=backbone, nc=21, 
             aux_classifier=aux_classifier, pad_input=True)


# FCN32-VGG16_BN
def fcn_vgg(nc=21, aux:bool=False):
  backbone = InterLayer(nets.vgg16_bn().features, {'23': 'aux', '30': 'out'})
  aux_classifier = FCNHead(ic=512, oc=nc) if aux else None
  return FCN(head_ic=512, backbone=backbone, nc=21, 
             aux_classifier=aux_classifier, pad_input=True)


# FCN32-GoogLeNet
def fcn_googlenet(nc=21, aux:bool=False):
  backbone = InterLayer(nets.googlenet(), {'inception4e': 'aux', \
                                           'inception5b': 'out'})
  aux_classifier = FCNHead(ic=1024, oc=nc) if aux else None
  return FCN(head_ic=1024, backbone=backbone, nc=21, 
             aux_classifier=aux_classifier, pad_input=True)



# =============================================================================
# FCN-16s
# =============================================================================

# FCN16-Alexnet
def fcn16_alexnet(nc=21, aux:bool=False):
  backbone = InterLayer(nets.alexnet().features, {'9': 'skip1', '12': 'out'})
  aux_classifier = FCNHead(ic=256, oc=nc) if aux else None
  return FCN(head_ic=256, backbone=backbone, nc=21, 
             aux_classifier=aux_classifier, pad_input=True)



# =============================================================================
# FCN-8s
# =============================================================================
