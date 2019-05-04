import unittest
import torch
import torch.nn as nn
from scratchai import *

class TestLearners(unittest.TestCase):
    
  def test_clf_test(self):
    # Check if the function is implemented
    utils.implemented(learners.clflearner, 'clf_test')

  def test_clf_train(self):
    # Check if the function is implemented
    utils.implemented(learners.clflearner, 'clf_train')

  def test_clf_fit(self):
    # Check if the function is implemented
    utils.implemented(learners.clflearner, 'clf_fit')

  def test_imp_mnist_cifar10(self):
    # Check if the function is implemented
    utils.implemented(learners.quicktrain, 'mnist')
    utils.implemented(learners.quicktrain, 'cifar10')
  
  @unittest.skipIf(torch.cuda.is_available() == False, 'no cuda')
  def test_train_mnist_cifar10(self):
    # Check if the function is implemented
    net = nets.lenet_mnist()
    learners.quicktrain.mnist(net, epochs=1)
    net = nets.lenet_cifar10()
    learners.quicktrain.cifar10(net, epochs=1)

  def test_adjust_lr(self):
    # Check if the function is implemented
    utils.implemented(learners.clflearner, 'adjust_lr')
