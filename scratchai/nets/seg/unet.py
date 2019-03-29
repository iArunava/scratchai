import torch
import torch.nn as nn

def conv(ic, oc, ks=3, s=1, p=0, bn=False):
    layers = [nn.Conv2d(ic, oc, kernel_size=ks, stride=s, padding=p, bias=bn]
    if norm: layers += [nn.BatchNorm2d(oc)]
    layers += [nn.ReLU(inplace=True)]
    return *layers

def mpool(ks=2, s=2, p=1):
    return nn.MaxPool2d(kernel_size=ks, stride=2, padding=1)

def uconv(ic, oc, ks=2, s=1, p=0, bn=False, act=False):
    layers = [nn.ConvTranspose2d(ic, oc, kernel_size=ks, stride=s, padding=p, bias=bn]
    if norm: layers += [nn.BatchNorm2d(oc)]
    if act: layers += [nn.ReLU(inplace=True)]
    return *layers

class UNet_EBlock(nn.Module):
    def __init__(self, ic, oc):
        uc = uconv(ic, oc)
        up = nn.Sequential(conv(ic, ic), conv(ic, oc), conv(oc, oc))
    def forward(self, i1, i2):
        '''
        Arguments:
        :: i1 - output of the layer below
        :: i2 - output from the side layer
        '''
        _, _, d1, d2 = (i2.shape - i1.shape) / 2
        _, _, h, w = i1.shape
        i2 = i2[:, :, d1:d1+h, d2:d2+w]
        return up(torch.cat((uc(i1), i2), dim=1))
        
class UNet(nn.Module):
    '''
    The UNet Architecture.
    Reproduced from paper: https://arxiv.org/pdf/1505.04597.pdf

    Arguments:
    :: ic - # of input channels
    :: nc - # of classes
    :: sd - # of filters for first convolution
    '''

    def __init__(self, ic, nc, sd):
        
        super().__init__()
        
        self.ic = ic; self.nc = nc
        
        ud1 = nn.Sequential(conv(ic, sdim), conv(sd, sd), conv(sd, sd), mpool())
        ud2 = nn.Sequential(conv(sd, sd), conv(sd*2, sd*2), conv(sd*2, sd*2), mpool())
        ud3 = nn.Sequential(conv(sd*2, sd*2), conv(sd*4, sd*4), conv(sd*4, sd*4), mpool())
        ud4 = nn.Sequential(conv(sd*4, sd*4), conv(sd*8, sd*8), conv(sd*8, sd*8), mpool())
        ud5 = nn.Sequential(conv(sd*8, sd*8), conv(sd*8, sd*16), conv(sd*16, sd*16), mpool())

        ue1 = UNet_EBlock(sd*16, sd*8)
        ue2 = UNet_EBlock(sd*8, sd*4)
        ue3 = UNet_EBlock(sd*4, sd*2)
        ue4 = UNet_EBlock(sd*2, sd)

        fconv = conv(sd, nc)

    def forward(self, x):
        o1 = ud1(x); o2 = ud2(x);
        o3 = ud3(x); o4 = ud4(x)
        o5 = ud5(x); o6 = ue1(o5, o4);
        o7 = ue(o6, o3); o8 = ue(o7, o2); 
        o9 = ue(o8, o1)

        return fconv(sd, nc)
