import torch
import torch.nn as nn

from .bnconv import bnconv

class Resblock(nn.Module):
 
 def __init__(self, ic:int, oc:int, stride:int=1, num_conv:int=2, 
       expansion:int=4, conv_first:bool=True, relu_after_add:bool=True, 
       eps:float=1e-5, momentum:float=0.1, inplace=True):
  '''
  This class defines the Residual Basic Block with n Conv Layers

  Arguments;
  - ic :: # of input channels
  - oc :: # of output channels for the final conv layers
  - downsample :: Whether this block is to downsample the input
  - expansion :: The expansion for the channels
         Default: 4
  - stride :: if downsample is True then the specified stride is used.
       Default: 2
  - conv_first :: Whether to apply conv before bn or otherwise.
         Default: True
  
  - num_conv :: # of conv layers in the block for the main branch
  '''
  
  super().__init__()
  
  assert(relu_after_add == True or relu_after_add == False)

  self.relu_after_add = relu_after_add
  self.downsample = True if stride > 1 or ic != oc else False
  oc_convi = oc // expansion
  
  layers = [bnconv(in_channels=ic,
            out_channels=oc_convi,
            kernel_size=3,
            padding=1,
            stride=stride,
            eps=eps,
            momentum=momentum,
            conv_first=conv_first,
            inplace=inplace)]
  
  for _ in range(num_conv-2):
    layers += [bnconv(in_channels=oc_convi,
              out_channels=oc_convi,
              kernel_size=3,
              padding=1,
              stride=1,
              eps=eps,
              momentum=momentum,
              conv_first=conv_first,
              inplace=inplace)]
  
  layers += [bnconv(in_channels=oc_convi,
            out_channels=oc,
            kernel_size=3,
            padding=1,
            stride=1,
            eps=eps,
            momentum=momentum,
            conv_first=conv_first,
            inplace=inplace,
            act=False)]
  
  self.main = nn.Sequential(*layers)

  if self.downsample:
    self.side = bnconv(in_channels=ic,
              out_channels=oc,
              kernel_size=3,
              padding=1,
              stride=stride,
              eps=eps,
              momentum=momentum,
              conv_first=conv_first,
              inplace=inplace,
              act=False)
  
  if self.relu_after_add:
    self.relu = nn.ReLU(inplace=inplace)
  
 def forward(self, x):

  x = self.main(x) + (self.side(x) if self.downsample else x)
  return self.relu(x) if self.relu_after_add else x

class Resnext(nn.Module):
 
 def __init__(self, ic:int, oc:int, stride:int=1, g:int=32, num_conv:int=2, 
       expansion:int=4, conv_first:bool=True, relu_after_add:bool=False, eps:float=1e-5, momentum:float=0.1, inplace=True, icf:int=1):
  '''
  This class defines the Residual Basic Block with n Conv Layers

  Arguments;
  :: ic         - # of input channels
  :: oc         - # of output channels for the final conv layers
  :: downsample - Whether this block is to downsample the input
  :: expansion  - The expansion for the channels
          Default: 4
  :: stride     - if downsample is True then the specified stride is used.
          Default: 2
  :: conv_first - Whether to apply conv before bn or otherwise.
         Default: True
  
  :: num_conv   - # of conv layers in the block for the main branch
  :: g          - # of groups
  :: icf        - Increase the number of input channels by icf in the 
          first conv.
  '''
  
  super().__init__()
  
  assert(relu_after_add == True or relu_after_add == False)

  self.relu_after_add = relu_after_add
  self.downsample = True if stride > 1 or ic != oc else False
  oc_1 = ic * icf
  
  layers = [bnconv(in_channels=ic,
            out_channels=oc_1,
            kernel_size=1,
            padding=0,
            stride=stride,
            eps=eps,
            momentum=momentum,
            conv_first=conv_first,
            inplace=inplace)]
  
  layers += [nn.Conv2d(in_channels=oc_1,
            out_channels=oc_1,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=g,
            bias=False)]
  
  layers += [bnconv(in_channels=oc_1,
            out_channels=oc,
            kernel_size=1,
            padding=0,
            stride=1,
            eps=eps,
            momentum=momentum,
            conv_first=conv_first,
            inplace=inplace,
            act=False)]
  
  self.main = nn.Sequential(*layers)

  if self.downsample:
    self.side = bnconv(in_channels=ic,
              out_channels=oc,
              kernel_size=3,
              padding=1,
              stride=stride,
              eps=eps,
              momentum=momentum,
              conv_first=conv_first,
              inplace=inplace,
              act=False)
  
  if self.relu_after_add:
    self.relu = nn.ReLU(inplace=inplace)
  
 def forward(self, x):
  
  x1 = self.main(x)
  print (x1.shape)
  x = self.main(x) + (self.side(x) if self.downsample else x)
  return self.relu(x) if self.relu_after_add else x

def res_stage(block:nn.Module, ic:int, oc:int, num_layers:int,
       stride:int=2, conv_first=False, inplace=False, lconv:int=2):
  
  '''

  Arguments:
  
  :: block - the block type to be stacked one upon another
  :: ic   - # of input channels
  :: oc  - # of output channels

  :: num_layers - # of blocks to be stacked
  :: stride     - Denotes the stride of the first conv block in the resnet block
          and the stride of the conv block in the side branch.
          If stride > 1 then downsample is triggered.
          Else the HxW is same before and after input and output.
  :: conv_first - Whether
          conv -> norm -> relu (True)
          relu -> norm -> conv (False)
  :: inplace    - Whether to perform inplace operations
  :: lconv      - # of conv in the main branch (resnet) of each stacked block
  '''

  layers = []
  layers += nn.ModuleList([block(ic=ic, oc=oc, stride=stride,
                  conv_first=conv_first, inplace=inplace,
                  num_conv=lconv)])

  layers.extend([block(ic=oc, oc=oc, conv_first=conv_first, 
            num_conv=lconv, inplace=inplace) for i in range (num_layers-1)])
  
  return nn.Sequential(*layers)
