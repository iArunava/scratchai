import unittest
import torch
import torch.nn as nn
import numpy as np
import random

from scratchai.trainers import metrics as M

class TestMetrics(unittest.TestCase):
  
  def test_miou(self):
    # TODO Add some more tests
    utils.implemented(M, 'miou')


  def test_confusion_matrix(self):
    
    for _ in range(random.randint(1, 10)):
      # Testing Valid Inputs
      nc = random.randint(2, 100)
      true = np.random.randint(0, nc, (3, 10, 10))
      pred = np.random.randint(0, nc, (3, 10, 10))
      cf = M.confusion_matrix(nc, true=true, pred=pred)
      
      self.assertTrue(isinstance(cf, np.ndarray), 'Output Type wrong!')
      self.assertEqual(cf.shape, (nc, nc), 'Out Shape wrong!')
      self.assertEqual(np.diag(cf).sum(), (true == pred).sum(), 'Nope!')
      
      for _ in range(10):
        c1, c2 = np.random.randint(0, nc), np.random.randint(0, nc)
        idx = np.logical_and((true == c1),  (pred == c2))
        self.assertEqual(cf[c1, c2], idx.sum(), 'Nope!')
    
      # Testing Invalid Inputs
      true = torch.randint(0, nc, (3, 10, 10))
      self.assertRaises(AssertionError,
                        lambda : M.confusion_matrix(nc, true=true, pred=true))
      self.assertRaises(AssertionError, lambda : \
                        M.confusion_matrix(nc-(nc-2), true=pred, pred=pred))

  def test_accuracy(self):
    # Stress Testing
    bs = 16; cls = 10
    for _ in range(torch.randint(0, 10, ())):
      pred = torch.rand(bs, cls)
      gt = torch.randint(0, cls, size=(bs,))
      top1 = (torch.argmax(pred, dim=1) == gt).float().sum()
      out = M.accuracy(pred, gt, topk=(1, 3, 5, 10))
      self.assertEqual(out[-1], 1, 'result bad!')
      self.assertEqual(out[0], top1 / bs, 'result bad!')
    
    # Explicit test
    pred = torch.Tensor([[0.6750, 0.5202, 0.0741, 0.0337, 0.8963],
                          [0.7856, 0.6455, 0.5330, 0.9970, 0.8105],
                          [0.2803, 0.4215, 0.6472, 0.4425, 0.7584],
                          [0.7624, 0.4732, 0.2856, 0.1151, 0.1639],
                          [0.8577, 0.2324, 0.3094, 0.7254, 0.2919],
                          [0.7253, 0.7911, 0.7737, 0.7527, 0.3939],
                          [0.8724, 0.5095, 0.5858, 0.3531, 0.8273],
                          [0.1239, 0.4820, 0.3762, 0.1830, 0.9889],
                          [0.9766, 0.8676, 0.7204, 0.8315, 0.0323],
                          [0.1487, 0.1719, 0.9073, 0.8438, 0.0470]])
                      
    # topk index        #1 #1 #1 #3 #3 #3 #5 #5 #3 #5
    gt = torch.Tensor([[4, 3, 4, 2, 2, 3, 3, 0, 3, 4]]).view(-1, 1).long()
    out = M.accuracy(pred, gt, topk=(1, 3, 5))
    self.assertTrue(out == [3/10, 7/10, 10/10], 'result bad!')

    self.assertRaises(Exception, lambda: M.accuracy(pred, gt, topk=(1)))
    self.assertRaises(Exception, lambda: M.accuracy(pred, gt, topk=1.))
