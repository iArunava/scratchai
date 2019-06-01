import torch
import torch.nn as nn
import unittest

from torchvision import transforms as T
from scratchai.datasets.labels import imagenet_labels
from scratchai import *

class TestOneCalls(unittest.TestCase):
  
  url_1 = 'https://cdn.instructables.com/FVI/FJPH/FWYHRVGC/FVIFJPHFWYHRVGC.'\
          'LARGE.jpg'
  lab_1 = 'rain barrel'

  url_2 = 'http://bradleymitchell.me/wp-content/uploads/2014/06/decompressed.'\
          'jpg'
  lab_2 = 'five'

  def test_classify(self):
    """
    This function ensures the classify function is working properly.
    """
    ns = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 
         'resnet152']
    
    # Check that url works
    for n in ns:
      pred, val = one_call.classify(TestOneCalls.url_1, nstr=n)
      self.assertTrue(pred == TestOneCalls.lab_1, 'Doesn\'t Work!')

    pred, val = one_call.classify(TestOneCalls.url_1)
    self.assertTrue(isinstance(pred, str), 'Doesn\'t Work!')
    self.assertTrue(pred == TestOneCalls.lab_1, 'Doesn\'t Work!')
    # TODO Check that path in local works
    
    # Checks for one_classify can take nn.Module as argument
    net = nets.resnet18()
    pred, val = one_call.classify(TestOneCalls.url_1, nstr=net)
    self.assertTrue(isinstance(pred, str), 'Doesn\'t Work!')
    self.assertTrue(pred == TestOneCalls.lab_1, 'Doesn\'t Work!')

    # Check that mnist works
    mnist_ns = ['lenet_mnist', 'alexnet_mnist']
    for n in mnist_ns:
      pred, val = one_call.classify(TestOneCalls.url_2, nstr=n, 
                                    trf='rz32_cc28_tt')
      pred, val = one_call.classify(TestOneCalls.url_2, nstr=n)

      self.assertTrue(isinstance(pred, str), 'Doesn\'t Work!')
      self.assertTrue(pred == TestOneCalls.lab_2, 'Doesn\'t Work!')

    self.assertRaises(AssertionError, lambda: one_call.classify(
            TestOneCalls.url_1, trf=imgutils.get_trf('tt_normmnist')))
    
    # Checks for one_classify and MNIST that it can take nn.Module as input
    # Given that gray is set to True for 2D images.
    net = nets.lenet_mnist()
    pred, val = one_call.classify(TestOneCalls.url_2, nstr=net, gray=1)
    self.assertTrue(isinstance(pred, str), 'Doesn\'t Work!')
    self.assertTrue(pred == TestOneCalls.lab_2, 'Doesn\'t Work!')

  def test_stransfer(self):
    """
    Ensures the stransfer function is working properly.
    """
    tt = T.ToTensor()
    imgshape = tt(imgutils.load_img(TestOneCalls.url_1)).shape
    simgshape = tt(one_call.stransfer(TestOneCalls.url_1, show=False)).shape
    self.assertTrue(imgshape == simgshape, 'doesn\'t look good')

  def test_attack(self):
    """
    Ensures the one_call.attack function is working as expected.
    """
    atks = [attacks.FGM, attacks.PGD, attacks.Semantic]
    for atk in atks:
      limgs, tlabl, plabl = one_call.attack(TestOneCalls.url_1, atk=atk, ret=1)
      self.assertFalse(tlabl == plabl, 'attack didn\t work')
      self.assertTrue(limgs[1].sum() != 0., 'attack didn\t work')
    # Noise attack not supported through one_call
    self.assertRaises(AssertionError, lambda: \
        one_call.attack(TestOneCalls.url_1, atk=attacks.Noise))
    # Label should be between 0 and 1000 (1000 excluded)
    self.assertRaises(AssertionError, lambda: \
        one_call.attack(TestOneCalls.url_1, atk=attacks.PGD, y=1000))

    
    y = torch.randint(0, 1000, ())
    limgs, tlabl, plabl = one_call.attack(TestOneCalls.url_1, atk=attacks.PGD, 
                                   y=y.item(), ret=True)
    self.assertTrue(plabl[0] == imagenet_labels[y.item()], 'not working!')
