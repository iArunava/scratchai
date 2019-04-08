"""
The Tests that needs to be performed on scratchai.nets
"""

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
        net = scratchai.nets.seg.enet.InitialBlock(3, 3)
        out = net(noise)
        self.assertEqual(list(out.shape), [2, 6, 2, 2], "out shape reduction not as it should"
                                                            " be.")
    def test_RDANeck(self):
        noise = torch.randn(2, 8, 4, 4)

        net = scratchai.nets.seg.enet.RDANeck(8, 8)
        out = net(noise)
        self.assertEqual(list(out.shape), [2, 8, 4, 4], "out shape reduction not as it should"
                                                            " be.")
        net = scratchai.nets.seg.enet.RDANeck(8, 9)
        out = net(noise)
        self.assertEqual(list(out.shape), [2, 9, 4, 4], "out shape reduction not as it should"
                                                            " be.")
    def test_DNeck_UNeck(self):
        noise = torch.randn(2, 8, 4, 4)
        noise2 = torch.randn(2, 8, 2, 2)

        net = scratchai.nets.seg.enet.DNeck(8, 8)
        out, idxs = net(noise)
        self.assertEqual(list(out.shape), [2, 8, 2, 2], "out shape reduction not as it should"
                                                            " be.")
        net = scratchai.nets.seg.enet.UNeck(8, 8)
        out = net(noise2, idxs)
        self.assertEqual(list(out.shape), [2, 8, 4, 4], "out shape reduction not as it should"
                                                            " be.")
        """
        net = scratchai.nets.seg.enet.DNeck(8, 9)
        out, idxs = net(noise)
        self.assertEqual(list(out.shape), [2, 9, 2, 2], "out shape reduction not as it should"
                                                            " be.")
        net = scratchai.nets.seg.enet.UNeck(8, 9)
        out = net(noise2, idxs)
        self.assertEqual(list(out.shape), [2, 8, 4, 4], "out shape reduction not as it should"
                                                            " be.")

        net = scratchai.nets.seg.enet.DNeck(8, 7)
        out, idxs = net(noise)
        self.assertEqual(list(out.shape), [2, 7, 2, 2], "out shape reduction not as it should"
                                                            " be.")
        net = scratchai.nets.seg.enet.UNeck(8, 7)
        out = net(noise2, idxs)
        self.assertEqual(list(out.shape), [2, 9, 4, 4], "out shape reduction not as it should"
                                                            " be.")
        """

    def test_enet(self):
        noise = torch.randn(2, 3, 256, 256)
        net = scratchai.ENet(4)
        out = net(noise)
        self.assertEqual(list(out.shape), [2, 4, 256, 256], "out shape reduction not as it should"
                                                            " be.")
        
if __name__ == '__name__':
    unittest.main()
