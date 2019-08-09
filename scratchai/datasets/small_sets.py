"""
This file contains classes to create dataset of smaller sizes for quick
Iterations.
"""
from torchvision.datasets import MNIST

class MNIST(MNIST):
  """
  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.data = self.data[:512]
    self.targets = self.targets[:512]
