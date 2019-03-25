import torch
import torch.nn as nn

from .bnconv import bnconv

class ResneXt(nn.Module):
  
  def __init__(self, ic:int, oc:int, stride:int=1, num_conv:int=2, 
               expansion:int=4, conv_first:bool=True, relu_after_add:bool=False, 
               eps:float=1e-5, momentum:float=0.1, inplace=True, icf:int=1):
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
    self.downsample = True if stride > 1 else False
    oc_1 = ic * icf
    
    layers = [bnconv(in_channels=ic,
                        out_channels=oc_1,
                        kernel_size=1,
                        padding=1,
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
                        eps=eps,
                        momentum=momentum,
                        groups=groups)]
    
    layers += [bnconv(in_channels=oc_1,
                        out_channels=oc,
                        kernel_size=1,
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

    x = self.main(x) + self.side(x) if self.downsample else x
    return self.relu(x) if self.relu_after_add else x
