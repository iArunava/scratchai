"""
The Tests that needs to be performed on scratchai.nets
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
import numpy as np

import scratchai
from scratchai import *

class TestUNet(unittest.TestCase):
  def test_paper(self):
    '''
    This module tests the UNet implementation to be the same
    as shown in the paper.
    '''
    noise = torch.randn(2, 3, 572, 572)
    # TODO Assert device is on cpu after initialization

    unet1 = scratchai.nets.UNet(3, 4, sos=False)
    out = unet1(noise)
    self.assertEqual(list(out.shape), [2, 4, 388, 388], "The out shape not same as in shape")

    unet2 = scratchai.nets.UNet(3, 4, sos=True)
    out = unet2(noise)
    self.assertEqual(list(out.shape), [2, 4, 572, 572], "The out shape not same as in shape")

  def test_conv(self):
    conv = scratchai.nets.seg.unet.conv(3, 3)[0]
    noise = torch.randn(2, 3, 4, 4)
    out = conv(noise)
    self.assertEqual(list(out.shape), [2, 3, 2, 2], "The out shape not same as in shape")

  def test_uconv(self):
    conv = scratchai.nets.seg.unet.uconv(3, 3)[0]
    noise = torch.randn(2, 3, 2, 2)
    out = conv(noise)
    self.assertEqual(list(out.shape), [2, 3, 4, 4], "The out shape not same as in shape")

  def test_ublock(self):
    '''
    Input: [N, C, H, W]
    Output: [N, C//2, (H*2) - 4, (W*2) - 4]
    '''
    net = scratchai.nets.seg.unet.UNet_EBlock(4)
    n1 = torch.randn(2, 4, 52, 52)
    n2 = torch.randn(2, 2, 136, 136)
    out = net(n1, n2)
    self.assertEqual(list(out.shape), [2, 2, 100, 100], "The out shape not same as in shape")

class TestENet(unittest.TestCase):

  def test_initial_block(self):
    noise = torch.randn(2, 3, 4, 4)
    net = scratchai.nets.seg.enet.InitialBlock(3, 6)
    out = net(noise)
    self.assertEqual(list(out.shape), [2, 6, 2, 2], "out shape reduction not as it should"
                              " be.")
  def test_RDANeck(self):
    noise = torch.randn(2, 8, 4, 4)
    
    # Check 1
    net = scratchai.nets.seg.enet.RDANeck(8, 8, device='cpu')
    out = net(noise)
    self.assertEqual(list(out.shape), [2, 8, 4, 4], "out shape reduction not as it should"
                              " be.")
    # Check 2
    net = scratchai.nets.seg.enet.RDANeck(8, 9, device='cpu')
    out = net(noise)
    self.assertEqual(list(out.shape), [2, 9, 4, 4], "out shape reduction not as it should"
                              " be.")
    # Check 3
    net = scratchai.nets.seg.enet.RDANeck(8, 8, aflag=True, device='cpu')
    out = net(noise)
    self.assertEqual(list(out.shape), [2, 8, 4, 4], "out shape reduction not as it should"
                              " be.")
  def test_DNeck_UNeck(self):
    noise = torch.randn(2, 8, 4, 4)
    noise2 = torch.randn(2, 8, 2, 2)
    
    # Check 1
    net = scratchai.nets.seg.enet.DNeck(8, 8, device='cpu')
    out, idxs = net(noise)
    self.assertEqual(list(out.shape), [2, 8, 2, 2], "out shape reduction not as it should"
                              " be.")
    net = scratchai.nets.seg.enet.UNeck(8, 8)
    out = net(noise2, idxs)
    self.assertEqual(list(out.shape), [2, 8, 4, 4], "out shape reduction not as it should"
                              " be.")
    
    # Check 2
    net = scratchai.nets.seg.enet.DNeck(8, 9, device='cpu')
    out, idxs = net(noise)
    self.assertEqual(list(out.shape), [2, 9, 2, 2], "out shape reduction not as it should"
                              " be.")
    # Check 3
    noise = torch.randn(2, 7, 4, 4)
    noise2 = torch.randn(2, 8, 2, 2)

    net = scratchai.nets.seg.enet.DNeck(7, 7, device='cpu')
    out, idxs = net(noise)
    self.assertEqual(list(out.shape), [2, 7, 2, 2], "out shape reduction not as it should"
                              " be.")
    net = scratchai.nets.seg.enet.UNeck(8, 7)
    out = net(noise2, idxs)
    self.assertEqual(list(out.shape), [2, 7, 4, 4], "out shape reduction not as it should"
                             " be.")
  
  def test_enet(self):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n1 = torch.randn(2, 3, 256, 256).to(device)

    net = scratchai.nets.ENet(4).to(device)

    o1 = net(n1)
    self.assertEqual(list(o1.shape), [2, 4, 256, 256], "out shape reduction not as it should"
                              " be.");
    del n1, o1
    n1 = torch.randn(2, 3, 360, 512).to(device)
    o1 = net(n1)
    self.assertEqual(list(o1.shape), [2, 4, 360, 512], "out shape reduction not as it should"
                              " be.")


class TestResnet(unittest.TestCase):
  
  def test_resnet_init(self):
    noise = torch.randn(2, 3, 224, 224)
    models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    for model in models:
      nc = np.random.randint(1, 1000)
      net = getattr(scratchai.nets, model)(nc=nc, pretrained=False)
      out = net(noise)
      # TODO Assert device is on cpu after initialization
      self.assertEqual(list(out.shape), [2, nc], "out shape not looking good")

      net = getattr(scratchai.nets, model)(pretrained=False)
      out = net(noise)
      self.assertEqual(list(out.shape), [2, 1000], "out shape not looking good")
      del net, out

  def test_resnet18_mnist(self):
    noise = torch.randn(2, 1, 28, 28)
    net = getattr(scratchai.nets, 'resnet18_mnist')()
    out = net(noise)
    # TODO Assert device is on cpu after initialization
    self.assertEqual(list(out.shape), [2, 1000], "out shape not looking good")
    del net, out
    

class TestITN(unittest.TestCase):
  
  n1 = torch.randn(2, 3, 256, 256)
  def test_itnst_(self):
    out = nets.ITN_ST_()(TestITN.n1)
    self.assertEqual(list(out.shape), [2, 3, 256, 256], "out shape not looking good")

  def test_itnst(self):
    out = nets.ITN_ST()(TestITN.n1)
    self.assertEqual(list(out.shape), [2, 3, 256, 256], "out shape not looking good")


class TestLenet(unittest.TestCase):
 
  def test_lenet(self):
    n1 = torch.randn(2, 3, 32, 32)
    out = nets.Lenet(11)(n1)
    # TODO Assert device is on cpu after initialization
    self.assertEqual(list(out.shape), [2, 11], "out shape not looking good")


class TestAlexnet(unittest.TestCase):
  def test_alexnet(self):
    n1 = torch.randn(2, 3, 224, 224)
    net = nets.alexnet(nc=11, pretrained=False); out = net(n1)
    self.assertEqual(list(out.shape), [2, 11], "out shape not looking good")
    out = nets.alexnet(pretrained=False)(n1)
    self.assertEqual(list(out.shape), [2, 1000], "out shape not looking good")

    conv_dict = {0 : [(11, 11), (4, 4), (2, 2)],
                 3 : [(5, 5), (1, 1), (2, 2)],
                 (6, 8, 10) : [(3, 3), (1, 1), (1, 1)]}
    for key, val in conv_dict.items():
      if isinstance(key, tuple):
        for k in key:
          self.assertEqual(net.net[k].kernel_size, val[0], 'not good!')
          self.assertEqual(net.net[k].stride, val[1], 'not good!')
          self.assertEqual(net.net[k].padding, val[2], 'not good!')
      else:
        self.assertEqual(net.net[key].kernel_size, val[0], 'not good!')
        self.assertEqual(net.net[key].stride, val[1], 'not good!')
        self.assertEqual(net.net[key].padding, val[2], 'not good!')
    for i in [1, 4, 7, 9, 11, 17, 20]:
      self.assertIsInstance(net.net[i], nn.ReLU, 'not good!')

  def test_alexnet_mnist(self):
    n1 = torch.randn(2, 1, 28, 28)
    out = nets.alexnet_mnist(pretrained=False)(n1)
    self.assertEqual(list(out.shape), [2, 10], "out shape not looking good")
    

class TestVGG(unittest.TestCase):
  
  def test_vgg(self):
    n1 = torch.randn(2, 3, 224, 224)
    ns = ['vgg{}', 'vgg{}_bn']
    for c in ['11', '13', '16', '19']:
      for n in ns:
        net = getattr(nets, n.format(c))()
        out = net(n1)
        self.assertEqual(list(out.shape), [2, 1000], 'Nope!')
      
class TestCommon(nn.Module):
  
  def test_flatten(self):
    n1 = torch.randn(13, 3, 4, 2)
    out = nets.Flatten()(n1)
    self.assertTrue(out.shape, [13, 3*4*2], 'out shape not okay')

  def test_debug(self):
    utils.implemented(nets, 'debug')


#############################################################
######## Weght Initializations
#############################################################

class WeightInit(unittest.TestCase):

  def test_weight_init(self):
    net = nets.resnet18(pretrained=False)
    # TODO Add more tests
    net.apply(init.xavier_normal)
    net.apply(init.xavier_uniform)
    net.apply(init.kaiming_normal)
    net.apply(init.kaiming_uniform)
    net.apply(init.msr_init)
