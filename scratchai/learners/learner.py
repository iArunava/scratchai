import torch
import torch.nn as nn
from tabulate import tabulate

__all__ = ['Learner']

class Learner(object):
    
    def __init__(self, net:nn.Module, loader=None, metrics:list=None, e:int=10):
        '''
        A Learner Object

        Arguments:
        :: model - The model to train
        :: loader - The DataLoader from where to get the data
        :: e - The number of epochs
        '''
        self.net = net
        self.loader = loader
        self.metrics = metrics
        self.e = e
        self.h = 512
        self.w = 512

    def fit(self):
        assert loader is not None
        # TODO
        pass
    
    def conv_out_size(self, net):
        kh, kw = net.kernel_size if type(net.kernel_size) == tuple else (net.kernel_size, net.kernel_size)
        sh, sw = net.stride if type(net.stride) == tuple else (net.stride, net.stride)
        ph, pw = net.padding if type(net.padding) == tuple else (net.padding, net.padding)

        self.h = (int) ((self.h - kh + (2 * ph)) / sh) + 1
        self.w = (int) ((self.w - kw + (2 * pw)) / sw) + 1
        return self.h, self.w

    def unet_eblock_out(self, net):
        self.h = (int) ((self.h * 2) - 4)
        self.w = (int) ((self.w * 2) - 4)
        return self.h, self.w
    
    def summary(self):
        layers = [['Input'], [(self.h, self.w)]]
        print (tabulate(layers))
        self._summary(self.net)

    def _summary(self, net):
        layers = []
        for m in net.children():
            temp = []
            if isinstance(m, nn.Sequential):
                self._summary(m)

            elif isinstance(m, nn.Conv2d):
                temp.append('Conv2d({}, {}, {})'.format(m.kernel_size, m.stride, m.padding))
                temp.append('{}'.format(self.conv_out_size(m)))

            elif str(m.__class__).split('.')[-1][:-2] == 'UNet_EBlock':
                temp.append('UNet_EBlock({}, {})'.format(m.uc.in_channels, m.uc.out_channels))
                temp.append('{}'.format(self.unet_eblock_out(m)))

            elif str(m.__class__).split('.')[-1][:-2] == 'MaxPool2d':
                temp.append('MaxPool2d({}, {}, {})'.format(m.kernel_size, m.stride, m.padding))
                temp.append('{}'.format(self.conv_out_size(m)))
            else:
                temp.append('ReLU')
                temp.append('{}'.format((self.h, self.w)))
            
            if len(temp) > 0:
                layers.append(temp)
        
        print (tabulate(layers))
