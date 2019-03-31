import torch
import torch.nn as nn

def conv(ic:int, oc:int, ks:int=3, s:int=1, p:int=0, norm:bool=False, act:bool=True):
    layers = [nn.Conv2d(ic, oc, kernel_size=ks, stride=s, padding=p, bias=not norm)]
    if norm: layers += [nn.BatchNorm2d(oc)]
    if act: layers += [nn.ReLU(inplace=True)]
    return layers

def mpool(ks:int=2, s:int=2, p:int=1):
    return nn.MaxPool2d(kernel_size=ks, stride=2, padding=1)

def uconv(ic:int, oc:int, ks:int=2, s:int=2, p:int=0, norm:bool=False, act:bool=False):
    layers = [nn.ConvTranspose2d(ic, oc, kernel_size=ks, stride=s, padding=p, bias=not norm)]
    if norm: layers += [nn.BatchNorm2d(oc)]
    if act: layers += [nn.ReLU(inplace=True)]
    return layers

class UNet_EBlock(nn.Module):
    def __init__(self, ic):
        super().__init__()
        self.uc = uconv(ic, ic//2)[0]
        self.up = nn.Sequential(*conv(ic, ic//2), *conv(ic//2, ic//2))

    def forward(self, i1, i2):
        '''
        Arguments:
        :: i1 - output of the layer below
        :: i2 - output from the side layer
        '''
        dc = self.uc(i1)
        h1, h2, w1, w2 = i2.shape[2], dc.shape[2], i2.shape[3], dc.shape[3]
        d1, d2 = (h1 - h2) // 2, (w1 - w2) // 2
        i2 = i2[:, :, d1:d1+h2, d2:d2+w2]
        return self.up(torch.cat((dc, i2), dim=1))
        
class UNet(nn.Module):
    '''
    The UNet Architecture.
    Reproduced from paper: https://arxiv.org/pdf/1505.04597.pdf

    Arguments:
    :: ic - # of input channels
    :: nc - # of classes
    :: sd - # of filters for first convolution
    '''

    def __init__(self, ic, nc, sd=64):
        
        super().__init__()
        
        self.ic = ic; self.nc = nc
        
        self.ud1 = nn.Sequential(*conv(ic, sd), *conv(sd, sd),  mpool())
        self.ud2 = nn.Sequential(*conv(sd, sd*2), *conv(sd*2, sd*2), mpool())
        self.ud3 = nn.Sequential(*conv(sd*2, sd*4), *conv(sd*4, sd*4), mpool())
        self.ud4 = nn.Sequential(*conv(sd*4, sd*8), *conv(sd*8, sd*8), mpool())
        self.ud5 = nn.Sequential(*conv(sd*8, sd*16), *conv(sd*16, sd*16), mpool())

        self.ue1 = UNet_EBlock(sd*16)
        self.ue2 = UNet_EBlock(sd*8)
        self.ue3 = UNet_EBlock(sd*4)
        self.ue4 = UNet_EBlock(sd*2)

        self.fconv = conv(sd, nc, ks=1, act=False)[0]

    def forward(self, x):
        o1 = self.ud1(x); o2 = self.ud2(o1)
        o3 = self.ud3(o2); o4 = self.ud4(o3)
        o5 = self.ud5(o4); 
	
        o6 = self.ue1(o5, o4); o7 = self.ue2(o6, o3); 
        o8 = self.ue3(o7, o2); o9 = self.ue4(o8, o1)

        return self.fconv(o9)
