import torch
import torch.nn as nn

def bnconv(in_channels:int, out_channels:int, kernel_size, stride,
             padding:int, bias:bool=False, eps:float=1e-5, 
             momentum:float=0.1, conv_first=True, act:str='relu', 
             nslope:float=0.2, norm:str='batch', inplace:bool=True):
    
    '''
    This is the BNConv module that helps in defining the 
    Conv->Norm->RelU
    RelU->Norm->Conv
    
    Arguments:
    - in_channels = # of input channels
    - out_channels = # of output channels
    - kernel_size = the kernel_size
    - stride - the stride
    - padding - the padding
    - bias = whether to add bias to Convolutional Layer
    - eps = the epsilon value
    - momentum = the momentum value
    - conv_first = If True: conv->norm->act
                   If False: act->norm->conv
                   Default: True
    - nslope = the slope if leaky relu is used.
    - norm = the norm type
    - inplace = The value for the inplace argument inactivation
                Default: True
    '''

    normc = out_channels if conv_first else in_channels
    
    if act == 'relu':
        act = nn.ReLU(inplace=inplace)
    elif act == 'leakyrelu':
        act = nn.LeakyReLU(negative_slope=nslope, inplace=inplace)
    elif act == False:
        act = None
    else:
        raise ('Activation value not understood')

    if norm == 'batch':
        norm = nn.BatchNorm2d(normc, eps=eps, momentum=momentum)
    elif norm == 'instance':
        norm = nn.InstanceNorm2d(normc, eps=eps, momentum=momentum)
    else:
        raise ('Norm value not understood')
    
    layers = [nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        stride=stride,
                        bias=bias),
                
              norm,
              act]
    
    if act is None:
        layers = layers[:2]

    net = nn.Sequential(*layers) if conv_first else nn.Sequential(*layers[::-1])

    return net
