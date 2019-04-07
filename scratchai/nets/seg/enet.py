"""
This test file contains all the tests related to the nets
"""

import torch
import torch.nn as nn

def mpool(ks:int=2, s:int=2, p:int=0):
    return nn.MaxPool2d(kernel_size=ks, stride=s, padding=p)

def conv(ic:int, oc:int, ks:int=3, s:int=1, p:int=0, d:int=1, norm:bool=True, act:bool=nn.PReLU):
    layers = [nn.Conv2d(ic, oc, kernel_size=ks, stride=s, padding=p, dilation=d, bias=not norm)]
    if norm: layers += [nn.BatchNorm2d(oc)]
    if act: layers += [act(inplace=True)]
    return layers

class InitialBlock(nn.Module):
    """
    This module is the initial block of the ENet architecture
    """

    def __init__(self, ic=3, oc=13):
        super().__init__()
        self.main = nn.Sequential(conv(3, 13, 3, 2, 1))
        self.side = mpool()
        self.prelu = nn.PRelU()
    
    def forward(self, x):
        return self.prelu(torch.cat((self.main(x), self.side(x)), dim=1))

class RDANeck(nn.Module):
    """
    This module implements the Regular, Dilated and Asymmetric Blocks

    Args:
        ic: # of in_channels
        oc: # of out_channels
        d:  Dilation rate. 
            Default - 1
        dflag: Flag to indicate whether to downsample the inputs or not.
               Default - False
        pratio: Ratio by which the channels are reduced and increased in the main branch
                Default - 4
        p: The dropout probability. 
           Default - 0.1
    """

    def __init__(self, ic, oc, d=1, aflag=False, pratio=4, p=0.1, act:nn.Module=nn.ReLU):
        super().__init__()
        
        self.cpad = oc - ic
        ks = (1, 5)
        pad = (0, 2)
        rd = int(oc // pratio)

        l = [conv(rd, rd, ks, 1, pad), conv(rd, rd, ks[::-1], 1, pad[::-1])] if aflag \
            else [conv(rd, rd, 3, s, 1, d)]
        
        # TODO Add dropout
        self.main = nn.Sequential(conv(ic, rd, 1, 1, 0), *l, conv(rd, oc, 1, 1, 0, act=None))
        self.act = act(inplace=True)

    def forward(self, x):
        ix = x
        x = self.main(x)
        zshape = x.shape; zshape[1] = self.cpad
        # TODO Add device for torch.zeros
        out = ix + x if self.cpad else torch.concat((ix, torch.zeros(*zshape)), dim=1) + x
        return self.act(out)

class DNeck(nn.Module):
    """
    This module implements the Downsampling Block

    Args:
        ic: # of in_channels
        oc: # of out_channels
        d:  Dilation rate. 
            Default - 1
        dflag: Flag to indicate whether to downsample the inputs or not.
               Default - False
        pratio: Ratio by which the channels are reduced and increased in the main branch
                Default - 4
        p: The dropout probability. 
           Default - 0.1
    """

    def __init__(self, ic, oc, d=1, pratio=4, p=0.1, act:nn.Module=nn.ReLU):
        super().__init__()

        rd = int(oc // pratio)

        self.main = nn.Sequential((conv(ic, rd, 1, 1, 0), conv(rd, rd, 3, 2, 1), 
                                   conv(rd, oc, 1, 1, 0, act=None)))
        self.act = act(inplace=True)

    def forward(self, x):
        ix = x
        x = self.main(x)
