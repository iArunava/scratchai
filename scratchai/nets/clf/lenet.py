"""
The LeNet.
Paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
"""

import torch
import torch.nn as nn

from scratchai.nets.common import Flatten
from scratchai.utils import load_from_pth
from scratchai.pretrained import urls

__all__ = ['Lenet', 'lenet_mnist', 'lenet_cifar10']

def conv(ic:int, oc:int, k:int=5):
  layers = [nn.Conv2d(ic, oc, kernel_size=5), nn.MaxPool2d(2), 
            nn.ReLU(inplace=True)]
  return layers

class Lenet(nn.Module):
  """
  Implements the Lenet module.
  Paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

  Arguments
  ---------
  nc : int
       # of classes.
  ic : int
       # of input channels
  ex : int
       Expansion factor
  inhw : int
         The height and width of the input image
         (assuming both are same)
  """
  def __init__(self, nc:int=10, ic:int=3, ex:int=1, inhw:int=32):
    super().__init__()
    # TODO Needs refactoring, as its hard coded for 32x32x3 and 28x28x1
    # Change the op a/c to the inhw
    op = 5 if inhw == 32 else 4 # else assumes input size is 28 (for MNIST)
    layers = [*conv(ic, 6*ex), *conv(6*ex, 16*ex), Flatten(), 
              nn.Linear(16*op*op*ex, 120*ex), nn.Linear(120*ex, 84*ex), 
              nn.Linear(84*ex, nc)]
    self.net = nn.Sequential(*layers)

  def forward(self, x): return self.net(x)


def lenet_mnist(pretrained=True, **kwargs):
  kwargs['ic'] = 1; kwargs['inhw'] = 28
  net = Lenet(**kwargs)
  if pretrained: 
    net.load_state_dict(load_from_pth(urls.lenet_mnist_url, 'lenet_mnist'))
  return net

def lenet_cifar10(pretrained=False, **kwargs):
  kwargs['ic'] = 3; kwargs['inhw'] = 32
  net = Lenet(**kwargs)
  if pretrained: 
    net.load_state_dict(load_from_pth(urls.lenet_mnist_url, 'lenet_cifar10'))
  return net
