import sys
import os
import unittest
import torch
import scratchai

class TestUNet(unittest.TestCase):
    def test_paper(self):
        '''
        This module tests the UNet implementation to be the same
        as shown in the paper.
        '''
        unet = scratchai.nets.UNet(3, 4)
        noise = torch.randn(2, 3, 572, 572)
        out = unet(noise)
        self.assertEqual(list(out.shape), [2, 4, 388, 388], "The out shape not same as in shape")

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
        net = scratchai.nets.seg.unet.UNet_EBlock(4)
        n1 = torch.randn(2, 4, 52, 52)
        n2 = torch.randn(2, 2, 136, 136)
        out = net(n1, n2)
        self.assertEqual(list(out.shape), [2, 2, 100, 100], "The out shape not same as in shape")

if __name__ == '__name__':
    unittest.main()
