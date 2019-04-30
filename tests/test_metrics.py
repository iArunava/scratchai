import unittest
import torch
import torch.nn as nn
import numpy as np

from scratchai import *

class TestMetrics(nn.Module):
  
  def test_miou(self):
    # TODO Add some more tests
    is_imp = getattr(learners, 'miou')
    if not callable(is_imp): raise NotImplementedError

  def test_accuracy(self):
    '''
    pred = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    gt = torch.Tensor([0, 1, 4, 2, 3, 5, 6, 7, 8, 9, 10])
    acc = scratchai.learners.accuracy(pred, gt)
    self.assertEqual(acc, 
    '''
    # TODO Add some more tests
    is_imp = getattr(learners, 'accuracy')
    if not callable(is_imp): raise NotImplementedError
