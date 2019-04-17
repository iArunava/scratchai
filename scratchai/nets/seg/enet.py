"""
ENet: A Deep Neural Network Architecture forReal-Time Semantic Segmentation

"""

import torch
import torch.nn as nn
import copy
import numpy as np

def mupool(ks:int=2, s:int=2, p:int=0):
    return nn.MaxUnpool2d(kernel_size=ks, stride=s, padding=p)

def mpool(ks:int=2, s:int=2, p:int=0, idxs:bool=False):
    return nn.MaxPool2d(kernel_size=ks, stride=s, padding=p, return_indices=idxs)

def conv(ic:int, oc:int, ks:int=3, s:int=1, p:int=0, d:int=1, norm:bool=True, act:nn.Module=nn.PReLU):
    layers = [nn.Conv2d(ic, oc, kernel_size=ks, stride=s, padding=p, dilation=d, bias=not norm)]
    if norm: layers += [nn.BatchNorm2d(oc)]
    if act: layers += [act() if act == nn.PReLU else act(inplace=True)]
    return layers

def uconv(ic:int, oc:int, ks:int=2, s:int=2, p:int=0, op:int=0, norm:bool=True, act:bool=False):
    layers = [nn.ConvTranspose2d(ic, oc, kernel_size=ks, stride=s, padding=p, output_padding=op, \
                                 bias=not norm)]
    if norm: layers += [nn.BatchNorm2d(oc)]
    if act: layers += [nn.ReLU(inplace=True)]
    return layers

class InitialBlock(nn.Module):
    """
    This module is the initial block of the ENet architecture
    """

    def __init__(self, ic=3, oc=16):
        super().__init__()
        self.main = nn.Sequential(*conv(ic, oc-3, 3, 2, 1, norm=False, act=None))
        self.side = mpool()
        self.norm = nn.BatchNorm2d(oc)
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        return self.prelu(self.norm(torch.cat((self.main(x), self.side(x)), dim=1)))

class RDANeck(nn.Module):
    """
    This module implements the Regular, Dilated and Asymmetric Blocks

    Args:
        ic: # of in_channels
        oc: # of out_channels
        d:  Dilation rate. 
            Default - 1
        aflag: Flag to indicate whether it is an asymetric block or not.
               Default - False
        pratio: Ratio by which the channels are reduced and increased in the main branch
                Default - 4
        p: The dropout probability. 
           Default - 0.1
    """

    def __init__(self, ic:int, oc:int, d:int=1, aflag:bool=False, pratio:int=4, 
                 p:float=0.1, act:nn.Module=nn.ReLU, device:str='cuda'):

        super().__init__()
        
        self.cpad = np.abs(oc - ic)
        ks = (1, 5)
        pad = (0, 2)
        rd = ic // pratio
        self.dev = torch.device('cuda' if torch.cuda.is_available() and device=='cuda' \
                                else 'cpu')
        
        l = [*conv(rd, rd, ks, 1, pad, norm=None, act=None), \
             *conv(rd, rd, ks[::-1], 1, pad[::-1])] if aflag \
            else [*conv(rd, rd, 3, 1, d, d)]
        
        # TODO Check if the position of dropout is correct
        self.main = nn.Sequential(*conv(ic, rd, 1, 1, 0), *l, *conv(rd, oc, 1, 1, 0, act=None),
                                  nn.Dropout2d(p))
        self.act = act(inplace=True)

    def forward(self, x):
        ix = x
        x = self.main(x)
        zshape = list(x.shape); zshape[1] = self.cpad
        # TODO Add device for torch.zeros
        out = torch.cat((ix, torch.zeros(*zshape, device=self.dev)), dim=1) + x if self.cpad else ix + x
        return self.act(out)

class DNeck(nn.Module):
    """
    This module implements the Downsampling Block

    Args:
        ic: # of in_channels
        oc: # of out_channels
        pratio: Ratio by which the channels are reduced and increased in the main branch
                Default - 4
        p: The dropout probability. 
           Default - 0.1
    """

    def __init__(self, ic, oc, p=0.1, pratio=4, act:nn.Module=nn.PReLU, device='cuda'):
        super().__init__()

        self.cpad = np.abs(oc - ic)
        rd = ic // pratio
        self.dev = torch.device('cuda' if torch.cuda.is_available() and device=='cuda' \
                                else 'cpu')

        self.main = nn.Sequential(*conv(ic, rd, 1, 1, 0), *conv(rd, rd, 3, 2, 1), 
                                   *conv(rd, oc, 1, 1, 0, act=None))
        self.mpool = mpool(idxs=True)
        self.act = act(inplace=True) if act == nn.ReLU else act()

    def forward(self, x):
        ix, idxs = self.mpool(x)
        x = self.main(x)
        zshape = list(x.shape); zshape[1] = self.cpad
        # TODO Add device for torch.zeros
        out = torch.cat((ix, torch.zeros(*zshape, device=self.dev)), dim=1) + x if self.cpad else ix + x 
        return self.act(out), idxs

class UNeck(nn.Module):
    """
    This module implements the Downsampling Block
    Args:
        ic: # of in_channels
        oc: # of out_channels
        pratio: Ratio by which the channels are reduced and increased in the main branch
                Default - 4
        p: The dropout probability. 
           Default - 0.1
    """

    def __init__(self, ic:int, oc:int, p:float=0.1, pratio:int=4, act:nn.Module=nn.ReLU):
        super().__init__()

        self.cpad = oc - ic
        rd = ic // pratio
        # TODO Check if the position of dropout is correct
        self.main = nn.Sequential(*conv(ic, rd, 1, 1, 0), *uconv(rd, rd, 3, 2, 1, 1),
                                   *conv(rd, oc, 1, 1, 0, act=None), nn.Dropout2d(p))
        
        self.conv = nn.Sequential(*conv(ic, oc, 1, 1, 0))
        self.mupool = mupool()
        self.act = act(inplace=True)

    def forward(self, x, idxs):
        ix = self.mupool(self.conv(x), idxs)
        x = self.main(x)
        return self.act(x + ix)

class ENet(nn.Module):
    """
    This is the implementation of the ENet architecture.
    Reproduced from: https://arxiv.org/pdf/1606.02147.pdf

    Args:
        nc: # of classes
    """

    def __init__(self, nc):
        super().__init__()

        self.nc = nc
        
        self.init = InitialBlock()

        self.d1 = DNeck(16, 64, 0.01)
        self.b1 = nn.Sequential(*[RDANeck(64, 64, p=0.01) for _ in range(4)])
        
        self.d2 = DNeck(64, 128)
        l = [RDANeck(128, 128), RDANeck(128, 128, d=2), RDANeck(128, 128, aflag=True), 
             RDANeck(128, 128, d=4), RDANeck(128, 128), RDANeck(128, 128, d=8), 
             RDANeck(128, 128, aflag=True), RDANeck(128, 128, d=16)]
        self.b2_3 = nn.Sequential(*l, *copy.deepcopy(l))

        self.u1 = UNeck(128, 64)
        self.b4 = nn.Sequential(*[RDANeck(64, 64, act=nn.ReLU) for _ in range(2)])

        self.u2 = UNeck(64, 16)
        self.b5 = nn.Sequential(RDANeck(16, 16, act=nn.ReLU), *uconv(16, nc, 3, 2, 1, 1))

    def forward(self, x):
        o1, idx1 = self.d1(self.init(x))
        o2, idx2 = self.d2(self.b1(o1))
        return self.b5(self.u2(self.b4(self.u1(self.b2_3(o2), idx2)), idx1))
