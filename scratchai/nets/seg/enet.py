"""
ENet: A Deep Neural Network Architecture forReal-Time Semantic Segmentation

Abstract from paper:
The  ability  to  perform  pixel-wise  semantic  segmentation  in  real-time  is  ofparamount importance in mobile applications. Recent deep neural networks aimedat this task have the disadvantage of requiring a large number of floating point oper-ations and have long run-times that hinder their usability. In this paper, we proposea novel deep neural network architecture named ENet (efficient neural network),created specifically for tasks requiring low latency operation. ENet is up to 18×faster, requires 75×less FLOPs, has 79×less parameters, and provides similar orbetter accuracy to existing models. We have tested it on CamVid, Cityscapes andSUN datasets and report on comparisons with existing state-of-the-art methods,and the trade-offs between accuracy and processing time of a network. We presentperformance measurements of the proposed architecture on embedded systems andsuggest possible software improvements that could make ENet even faster.
"""

import torch
import torch.nn as nn

def mupool(ks:int=2, s:int=2, p:int=0):
    return nn.MaxUnpool2d(kernel_size=ks, stride=s, padding=p)

def mpool(ks:int=2, s:int=2, p:int=0, idxs:bool=False):
    return nn.MaxPool2d(kernel_size=ks, stride=s, padding=p, return_indices=idxs)

def conv(ic:int, oc:int, ks:int=3, s:int=1, p:int=0, d:int=1, norm:bool=True, act:bool=nn.PReLU):
    layers = [nn.Conv2d(ic, oc, kernel_size=ks, stride=s, padding=p, dilation=d, bias=not norm)]
    if norm: layers += [nn.BatchNorm2d(oc)]
    if act: layers += [act(inplace=True)]
    return layers

def uconv(ic:int, oc:int, ks:int=2, s:int=2, p:int=0, op:int=0, norm:bool=False, act:bool=False):
    layers = [nn.ConvTranspose2d(ic, oc, kernel_size=ks, stride=s, padding=p, output_padding=op, \
                                 bias=not norm)]
    if norm: layers += [nn.BatchNorm2d(oc)]
    if act: layers += [nn.ReLU(inplace=True)]
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
        aflag: Flag to indicate whether it is an asymetric block or not.
               Default - False
        pratio: Ratio by which the channels are reduced and increased in the main branch
                Default - 4
        p: The dropout probability. 
           Default - 0.1
    """

    def __init__(self, ic:int, oc:int, d:int=1, aflag:bool=False, pratio:int=4, 
                 p:float=0.1, act:nn.Module=nn.ReLU):

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
        pratio: Ratio by which the channels are reduced and increased in the main branch
                Default - 4
        p: The dropout probability. 
           Default - 0.1
    """

    def __init__(self, ic, oc, p=0.1, pratio=4, act:nn.Module=nn.PReLU):
        super().__init__()

        self.cpad = oc - ic
        rd = int(oc // pratio)
        self.main = nn.Sequential((conv(ic, rd, 1, 1, 0), conv(rd, rd, 3, 2, 1), 
                                   conv(rd, oc, 1, 1, 0, act=None)))
        self.act = act(inplace=True)
        self.mpool = mpool()
        self.act = act(inplace=True)

    def forward(self, x):
        ix, idxs = self.mpool(x)
        x = self.main(x)
        zshape = x.shape; zshape[1] = self.cpad
        # TODO Add device for torch.zeros
        out = ix + x if self.cpad else torch.concat((ix, torch.zeros(*zshape)), dim=1) + x
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
        rd = oc // pratio
        # TODO Dropout
        self.main = nn.Sequential((conv(ic, rd, 1, 1, 0), uconv(rd, rd, 3, 2, 1), 
                                   conv(rd, oc, 1, 1, 0, act=None)))
        
        self.conv = conv(ic, oc, 1, 1, 0)
        self.mpool = mupool()
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
        
        # ENet Architec
        self.init = InitialBlock()

        self.d1 = DNeck(16, 64, 0.01)
        self.b1 = nn.Sequential(*[RDANeck(64, 64, p=0.01) for _ in range(4)])
        
        self.d2 = DNeck(64, 128)
        self.b2_3 = nn.Sequential(*[RDANeck(128, 128), RDANeck(128, 128, d=2), 
                                RDANeck(128, 128, aflag=True), RDANeck(128, 128, d=4),
                                RDANeck(128, 128), RDANeck(128, 128, d=8), RDANeck(128, 128, aflag=True),
                                RDANeck(128, 128, d=16) for _ in range(2)])

        self.u1 = UNeck(128, 64)
        self.b4 = nn.Sequential(*[RDANeck(64, 64, act=nn.ReLU) for _ in range(2)])

        self.u2 = UNeck(64, 16)
        self.b5 = nn.Sequential(RDANeck(16, 16, act=nn.ReLU), uconv(16, nc, 3, 2, 1, 1))

    def forward(self, x):
        o1, idx1 = self.d1(self.init(x))
        o2, idx2 = self.d2(self.b1(o1))
        return self.b5(self.u2(self.b4(self.u1(self.b2_3(x), idx2)), idx1))
