import sys
import os
import unittest
import torch
from .. import scratchai

class TestUNet(unittest.TestCase):
    def test_simple(self):
        unet = scratchai.nets.unet(3, 4)
        noise = torch.randn(2, 3, 512, 512)
        out = unet(noise)
        self.assertEqual(list(noise.shape), [2, 4, 512, 512], "The out shape not same \
                                                               as in shape")

if __name__ == '__name__':
    print (__file__, __name__, __package__)
    unittest.main()
