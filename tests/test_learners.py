import unittest
import torch.nn as nn
from scratchai import *

class TestLearners(unittest.TestCase):
    
  def test_clf_test(self):
    # Check if the function is implemented
    utils.check_if_implemented(learners.clflearner, 'clf_test')

  def test_clf_train(self):
    # Check if the function is implemented
    utils.check_if_implemented(learners.clflearner, 'clf_train')

  def test_clf_fit(self):
    # Check if the function is implemented
    utils.check_if_implemented(learners.clflearner, 'clf_fit')

  def test_train_mnist(self):
    # Check if the function is implemented
    utils.check_if_implemented(learners.clflearner, 'train_mnist')

  def test_adjust_lr(self):
    # Check if the function is implemented
    utils.check_if_implemented(learners.clflearner, 'adjust_lr')
