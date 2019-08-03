"""
Loader to create dummy data to quickly test GANs
"""

import torch
from torchvision.datasets import VisionDataset


__all__ = ['GANDummyData']


class GANDummyData(VisionDataset):
  """
  Loader to create Dummy Data to check working of GANs

  Arguments
  ---------
  """
  
  # TODO Using root and transform even though its not required here.
  # as these arguments are required in VisionDataset class.
  def __init__(self, root='./', transform=None):
    super().__init__(root, transform)

  def __getitem__(self, index):
    img = torch.randn(3, 16, 16)
    labl = torch.randn(1)
    return img, labl

  def __len__(self):
    # TODO Just an arbitrary small number to quickly check if everything is working or not.
    return 6
