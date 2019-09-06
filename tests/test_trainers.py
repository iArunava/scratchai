import unittest
import torch
import torch.nn as nn
import random
import numpy as np

from torchvision import transforms as T
import torchvision.transforms.functional as F
from random import randint

from scratchai import *
from scratchai.trainers import transforms as ST

class TestLearners(unittest.TestCase):
    
  def test_imp_mnist_cifar10(self):
    # Check if the function is implemented
    utils.implemented(trainers.quicktrain, 'mnist')
    utils.implemented(trainers.quicktrain, 'cifar10')
  
  @unittest.skipIf(torch.cuda.is_available() == False, 'no cuda')
  def test_train_mnist_cifar10(self):
    # Check if the function is implemented
    net = nets.lenet_mnist()
    trainers.quicktrain.mnist(net, epochs=1)
    net = nets.lenet_cifar10()
    trainers.quicktrain.cifar10(net, epochs=1)


class TestTransforms(unittest.TestCase):
  
  def test_random_crop(self):
    for _ in range(randint(1, 10)):
      rh, rw = randint(100, 200),randint(100, 200)
      gt_shape = [rh, rw]

      rc = ST.RandomCrop((rw, rh))
      n1 = T.ToPILImage()(torch.randn(3, randint(200, 400), randint(200, 400)))
      n2 = T.ToPILImage()(torch.randn(3, randint(200, 400), randint(200, 400)))

      out1, out2 = rc(n1, n2)
      out1_shape, out2_shape = list(out1.size), list(out2.size)
      n1_shape, n2_shape = list(n1.size), list(n2.size)

      self.assertEqual(out1_shape, gt_shape, 'Nope!')
      self.assertEqual(out2_shape, gt_shape, 'Nope!')
    
    # Test to confirm that RandomCrop can take size as type int
    # TODO Repitive code try to club the above test and the below
    # into one.
    out_size = randint(100, 200)
    gt_shape = [out_size, out_size]

    rc = ST.RandomCrop(out_size)
    n1 = T.ToPILImage()(torch.randn(3, randint(200, 400), randint(200, 400)))
    n2 = T.ToPILImage()(torch.randn(3, randint(200, 400), randint(200, 400)))

    out1, out2 = rc(n1, n2)
    out1_shape, out2_shape = list(out1.size), list(out2.size)
    self.assertEqual(out1_shape, gt_shape, 'Nope!')
    self.assertEqual(out2_shape, gt_shape, 'Nope!')

  
  def test_random_horizontal_flip(self):
    for _ in range(randint(1, 10)):
      prob = random.random()
      n1 = T.ToPILImage()(torch.randn(3, randint(200, 400), randint(200, 400)))
      n2 = T.ToPILImage()(torch.randn(3, randint(200, 400), randint(200, 400)))
      n1_out, n2_out = ST.RandomHorizontalFlip(prob)(n1, n2)
      
      either_1_1 = np.all(np.asarray(n1) == np.asarray(n1_out))
      either_1_2 = np.all(np.asarray(F.hflip(n1)) == np.asarray(n1_out))
      either_2_1 = np.all(np.asarray(n2) == np.asarray(n2_out))
      either_2_2 = np.all(np.asarray(F.hflip(n2)) == np.asarray(n2_out))
      
      self.assertTrue((either_1_1 or either_1_2), 'Nope!')
      self.assertTrue((either_2_1 or either_2_2), 'Nope!')
      
  def test_pad_if_smaller(self):

    for _ in range(randint(1, 10)):
      h, w = randint(10, 100), randint(10, 100)
      n1 = T.ToPILImage()(torch.randn(3, h, w))
      _oh, _ow = randint(10, 100), randint(10, 100)
      oh, ow = max(_oh, h), max(_ow, w)
      n2 = ST.pad_if_smaller(n1, (_ow, _oh), fill=0)
      self.assertEqual(n2.size, (ow, oh), 'Shape not equal!')
      # TODO Test to check its filled correctly!
