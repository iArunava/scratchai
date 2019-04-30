import torch
import torch.nn as nn
import unittest
import scratchai
from scratchai.one_call import *

class TestOneCalls(unittest.TestCase):
  
  url_1 = 'https://cdn.instructables.com/FVI/FJPH/FWYHRVGC/FVIFJPHFWYHRVGC.LARGE.jpg'
  lab_1 = 'rain barrel'

  url_2 = 'http://bradleymitchell.me/wp-content/uploads/2014/06/decompressed.jpg'
  lab_2 = 'five'

  def test_classify(self):
    """
    This function ensures the classify function is working properly.
    """
    # Check that url works.
    pred = classify(TestOneCalls.url_1)
    self.assertTrue(isinstance(pred, str), 'Doesn\'t Work!')
    self.assertTrue(pred == TestOneCalls.lab_1, 'Doesn\'t Work!')

    # TODO Check that path in local works

    # Check that mnist works
    pred = classify(TestOneCalls.url_2, nstr='lenet_mnist', trf='rz32_cc28_tt')
    self.assertTrue(isinstance(pred, str), 'Doesn\'t Work!')
    self.assertTrue(pred == TestOneCalls.lab_2, 'Doesn\'t Work!')
