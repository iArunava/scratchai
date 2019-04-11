import scratchai
import torch
import torch.nn as nn
import unittest

class TestAttacks(unittest.TestCase):
    
    def test_noise_atk(self):
        """
        Tests to check that the Noise Attack works
        """
        noise = torch.randn(1, 3, 4, 4)
        net = nn.Conv2d(3, 3, 3, 2, 1)
        atk = scratchai.attacks.Noise(net, noise)
        
