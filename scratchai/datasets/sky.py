from torchvision.datasets import VisionDataset

class SkyDataset(VisionDataset):
  """
  Loader for the Sky Segmentation Dataset.

  Arguments
  ---------

  """
  def __init__(self, root, image_set='train', download=True, transforms=None):
