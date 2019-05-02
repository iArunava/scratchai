import unittest
import torch
import torch.nn as nn

from scratchai import *

class WeightInit(unittest.TestCase):
  
  def test_xavier_uniform(self):
    net = nets.lenet_mnist()
    # TODO Add more tests
    net.apply(init.xavier_normal)
    net.apply(init.xavier_uniform)
    net.apply(init.kaiming_normal)
