import torch
import torch.nn as nn

from .BNConv import BNConv

class ResBlock(nn.Module):
  
  def __init__(self, inc:int, outc:int, stride:int=1, num_conv:int=2, 
               expansion:int=4, conv_first:bool=True, relu_after_add:bool=True, 
               eps:float=1e-5, momentum:float=0.1, inplace=True):
    '''
    This class defines the Residual Basic Block with 2 Conv Layers

    Arguments;
    - inc :: # of input channels
    - outc :: # of output channels for the final conv layers
    - downsample :: Whether this block is to downsample the input
    - expansion :: The expansion for the channels
                  Default: 4
    - stride :: if downsample is True then the specified stride is used.
               Default: 2
    - conv_first :: Whether to apply conv before bn or otherwise.
                   Default: True
    
    - num_conv :: # of conv layers in the block for the main branch
    '''
    
    super(ResidualBlock, self).__init__()
    
    assert(relu_after_add == True or relu_after_add == False)

    self.relu_after_add = relu_after_add
    self.downsample = True if stride > 1 else False
    oc_convi = outc // expansion
    
    layers = [BNConv(in_channels=inc,
                        out_channels=oc_convi,
                        kernel_size=3,
                        padding=1,
                        stride=stride,
                        eps=eps,
                        momentum=momentum,
                        conv_first=conv_first,
                        inplace=inplace)]
    
    for _ in range(num_conv-2):
        layers += [BNConv(in_channels=oc_convi,
                            out_channels=oc_convi,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            eps=eps,
                            momentum=momentum,
                            conv_first=conv_first,
                            inplace=inplace)]
    
    layers += [BNConv(in_channels=oc_convi,
                        out_channels=outc,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        eps=eps,
                        momentum=momentum,
                        conv_first=conv_first,
                        inplace=inplace)]
    
    self.main = nn.Sequential(*layers)

    if self.downsample:
        self.side = BNConv(in_channels=inc,
                            out_channels=outc,
                            kernel_size=3,
                            padding=1,
                            stride=stride,
                            eps=eps,
                            momentum=momentum,
                            conv_first=conv_first,
                            inplace=inplace)
    
    if self.relu_after_add:
        self.relu = nn.ReLU(inplace=True)
    
  def forward(self, x):

    x = self.main(x) + self.side(x) if self.downsample else x
    return self.relu(x) if self.relu_after_add else x


def res_stage(block:nn.Module, inc:int, outc:int, num_layers:int,
                     stride:int=2, conv_first=False, inplace=False):
    '''
    '''
    layers = []
    layers += nn.ModuleList([block(inc=inc, outc=outc, stride=stride,
                                    conv_first=conv_first, inplace=inplace)])

    layers.extend([block(inc=outc, outc=outc,
                                conv_first=conv_first, inplace=inplace)
                                for i in range (num_layers-1)])
    
    return nn.Sequential(*layers)
