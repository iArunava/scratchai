"""
Contains some commonly used functions b/w all modules.
"""

import torch
import torch.nn as nn

from collections import OrderedDict

from scratchai import nets


__all__ = ['InterLayer', 'Flatten', 'Debug']


class InterLayer(nn.Module):
  """
  This class implements a Network that helps to get intermediate 
  layers from other Networks.

  Arguments
  ---------
  net           : nn.Module, str
                  The net from which to return the intermediate layers of.
  return_layers : list
                  Whose each element contains the details of the layer
                  from which to return the results.
                  So, for each element in return_layer this module returns
                  a result. And each element in the return_layers is
                  a pattern which indicates which layer to take the output
                  from. 
                  More on what type patterns the return_layers can take:
                  - 'key' : The usual key which comes from `named_modules()`
                  - 'b_x' : Output of the layer which comes just before layer
                            'x'. `x` should be a layer name from torch.nn.
                            Eg: `x` can be `AdaptiveAvgPool2d`
                  
                  Note: For now, it just supports the above defined patterns.

  Returns
  -------
  out : list
  """
  def __init__(self, net, return_layers):
    super().__init__()
    #assert type(net) in [str, nn.Module, nn.Sequential]
    # TODO Remove this print after there is a nice check that this sorted thing
    # is indeed the case.
    print ('[INFO] The layers in return_layers needs to be in a sorted fashion'\
           'Eg: ["skip1", "aux1", "skip2"], but not ["skip2", "aux1", "skip1"]')
    if isinstance(net, str): net = getattr(nets, net)()
    self.net = net
    self.return_layers = return_layers
    self.find_ocs(self.return_layers)
  
  def forward(self, x):
    out = OrderedDict()
    for name, layer in self.net.named_children():
      x = layer(x)
      if name in self.return_layers:
        out[self.return_layers[name]] = x
      
      if len(self.return_layers) == len(out): break
    return out
  

  def get_ocs(self, type_out):
    type_out = type_out.lower()
    if type_out.startswith('skip'): return self.skip_outs
    elif type_out.startswith('aux') : return self.aux_outs
    elif type_out.startswith('out') : return self.outs
    else: raise NotImplementedError('The type is not present')


  def find_ocs(self, layer_names):
    # TODO Optimize this function. No need to go over all the children.
    # NOTE Major assumption here: Only *nn.Conv2d* layers change the number
    # of channels. Probably this is not true.

    self.skip_outs = OrderedDict()
    self.aux_outs = OrderedDict()
    self.outs = OrderedDict()

    # NOTE The following line of code is wrong if inputs are grayscale.
    curr_oc = 3
    for name, layer in self.net.named_children():
      if isinstance(layer, nn.Conv2d):
        curr_oc = layer.out_channels
      if name in layer_names:
        key = layer_names[name]
        if key.startswith('skip'):
          self.skip_outs[layer_names[name]] = curr_oc
        elif key.startswith('aux'):
          self.aux_outs[layer_names[name]] = curr_oc
        else:
          self.outs[layer_names[name]] = curr_oc
      
      # Breaking out if all the required layers are found.
      if len(out) == len(layer_names): break

    # This reversing is necessary. For working of the FCN Model.
    #out = OrderedDict(reversed(list(out.items())))


class Flatten(nn.Module):
  def forward(self, x):
    return x.reshape(x.size(0), -1)


class Debug(nn.Module):
  def __init__(self, debug_value=None):
    super().__init__()
    self.debug_value = debug_value
  def forward(self, x):
    if self.debug_value: print (self.debug_value, end='')
    print (x.shape)
    return x
