"""
FCN - Fully Convolutional Neural Networks
Paper: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from copy import deepcopy

from scratchai import nets
from scratchai.imgutils import center_crop
from scratchai.nets.common import InterLayer
from scratchai.utils import bilinear_kernel
from scratchai.init import zero_init


__all__ = ['FCNHead', 'fcn_alexnet', 'fcn_vgg', 'fcn_googlenet', 
           'fcn16_alexnet', 'fcn8_alexnet']


def conv(ic:int, oc:int, ks:int):
  layers = []
  layers.append(nn.Conv2d(ic, oc, ks, 1, 0))
  layers.append(nn.ReLU(inplace=True))
  #layers.append(nn.Dropout2d(p=0.5))
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
  def __init__(self, ic:int, oc:int=21, dconv_ks:int=64, dconv_s:int=32,
               expand:int=12):
    super().__init__()
    
    self.inter_convs = 2 << (expand-1)
    self.net = nn.Sequential(
                  *conv(ic, self.inter_convs, 6), 
                  #*conv(self.inter_convs, self.inter_convs, 1),
                  nn.Conv2d(self.inter_convs, oc, 1, 1, 0),
                  #nn.ConvTranspose2d(oc, oc, dconv_ks, dconv_s, bias=False),
              )
    
    #self.net.apply(zero_init)
    #self.net[-1].weight.data.copy_(bilinear_kernel(oc, oc, dconv_ks))
    #self.net[-1].weight.requires_grad_(False)

  def forward(self, x): 
    x = self.net(x)
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

  skips          : int
                   The number of skip connections the net will have.
  """
  def __init__(self, head_ic:int, nc=21, backbone=None, aux_classifier=None,
               pad_input:bool=False, expand:int=12):
    super().__init__()

    dconv_ks = 64; dconv_s = 32
    self.pad_input = pad_input
    #self.backbone = backbone
    #self.backbone = nn.Conv2d(3, 256, 5, 3, 0)
    self.backbone = nn.Sequential(nn.Conv2d(3, 256, 1, 1, 0), nn.ReLU())#, nn.Conv2d(256, nc, 1, 1, 0))
    self.aux_classifier = aux_classifier
    """
    # Creating Extra Convolutional Layers as required.
    skips = deepcopy(self.backbone.return_layers)
    # TODO Think of a way to not reverse the dict twice.
    skips = dict((v, k) for k, v in skips.items())
    skips.pop('out', None); skips.pop('aux', None)
    skips = dict((v, k) for k, v in skips.items())
    self.skip_dicts = OrderedDict()
    num_skips = len(skips)
    if num_skips > 0:
      channels_dict = self.backbone.get_oc_for(skips)
      for ii, (key, val) in enumerate(channels_dict.items()):
        if (num_skips-1) == ii:
          dfactor = 2 << ii
          dc_ks, dc_s = (dconv_ks//dfactor, dconv_s//dfactor)
        else:
          dc_ks, dc_s = (4, 2)
        
        setattr(self, str(key)+str(ii), nn.Conv2d(val, nc, 1, 1, 0))
        setattr(self, str(key)+str(ii+1), nn.ConvTranspose2d(nc, nc, dc_ks, 
                                                             dc_s, bias=False))
        self.skip_dicts[key] = [getattr(self, str(key)+str(ii)),
                                getattr(self, str(key)+str(ii+1))]
      dconv_ks, dconv_s = 4, 2

    """
    self.fcn_head = FCNHead(head_ic, nc, dconv_ks, dconv_s, expand=expand)


  def forward(self, x):
    x_shape = x.shape[-2:]
    if self.pad_input:
      x = F.pad(x, (100, 100, 100, 100), mode='constant', value=0)
    
    out = OrderedDict()
    features_out = self.backbone(x)
    
    if self.aux_classifier is not None and 'aux' in features_out:
      # TODO Since we removed shape from fcn_head, aux will fail
      # if the input is padded, as is the case in general. as the output shape
      # and the target shape will differ.
      raise Exception('You poked the bear!')
      out['aux'] = self.aux_classifier(features_out['aux'])
    """
    out['out'] = self.fcn_head(features_out['out'])

    if len(self.skip_dicts) > 0:
      sout = out['out']
      for key, val in self.skip_dicts.items():
        curr_skip = val[0](features_out[key])
        curr_x = center_crop(curr_skip, sout.shape) + sout
        sout = val[1](curr_x)
      out['out'] = sout

    # Cropping the image to the required size (as mentioned by shape)
    out['out'] = center_crop(out['out'], x_shape)

    return out['out']
    """
    # TODO Remove the below lines
    features_out = self.fcn_head(features_out)
    out = F.interpolate(features_out, size=(360, 360), mode='bilinear')
    #print (out.shape); input()
    return out
    return center_crop(out, x_shape)



def get_fcn(nc, aux, features, return_layers:dict, head_ic, **kwargs):
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
  aux_classifier = FCNHead(ic=head_ic, oc=nc) if aux else None
  return FCN(head_ic=head_ic, backbone=backbone, nc=nc, 
             aux_classifier=aux_classifier, pad_input=True, **kwargs)



# =============================================================================
# FCN-32s
# =============================================================================

# FCN32-Alexnet
def fcn_alexnet(nc=21, aux:bool=False, **kwargs):
  return_layers = {'12': 'out'}
  if aux: return_layers['9'] = 'aux'
  return get_fcn(nc, aux, nets.alexnet().features, return_layers, 256, **kwargs)


# FCN32-VGG16_BN
def fcn_vgg(nc=21, aux:bool=False, **kwargs):
  return_layers = {'30': 'out'}
  if aux: return_layers['23'] = 'aux'
  return get_fcn(nc, aux, nets.vgg16_bn().features, return_layers, 512, **kwargs)


# FCN32-GoogLeNet
def fcn_googlenet(nc=21, aux:bool=False, **kwargs):
  return_layers = {'inception5b': 'out'}
  if aux: return_layers['inception4e'] = 'aux'
  return get_fcn(nc, aux, nets.googlenet(), return_layers, 1024, **kwargs)



# =============================================================================
# FCN-16s
# =============================================================================

# FCN16-Alexnet
def fcn16_alexnet(nc=21, aux:bool=False, **kwargs):
  return_layers = {'5': 'skip1', '12': 'out'}
  if aux: return_layers['9'] = 'aux'
  return get_fcn(nc, aux, nets.alexnet().features, return_layers, 256, **kwargs)



# =============================================================================
# FCN-8s
# =============================================================================
# FCN8-Alexnet
def fcn8_alexnet(nc=21, aux:bool=False, **kwargs):
  return_layers = {'2': 'skip2', '5': 'skip1', '12': 'out'}
  if aux: return_layers['9'] = 'aux'
  return get_fcn(nc, aux, nets.alexnet().features, return_layers, 256, **kwargs)
