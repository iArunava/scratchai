import os
import cv2

from PIL import Image
from torchvision.datasets import VisionDataset
from glob import glob

from scratchai.utils import download_and_extract


class SkyDataset(VisionDataset):
  """
  Loader for the Sky Segmentation Dataset.

  Arguments
  ---------

  """
  url = 'https://www.ime.usp.br/~eduardob/datasets/sky/sky.zip'

  def __init__(self, root='./', image_set='train', download=True, 
               transforms=None, target_transform=None):
    
    super().__init__(root, transforms)

    self.root = os.path.abspath(root)
    self.transforms = transforms
    self.target_transform = target_transform

    if download: download_and_extract(self.url, self.root)
    
    self.image_dir = os.path.join(self.root, 'sky/data/')
    self.mask_dir  = os.path.join(self.root, 'sky/groundtruth/')

    self.image_files = glob(self.image_dir + '/*')
    self.gd_files    = glob(self.mask_dir + '/*')


  def __getitem__(self, index):
    img = Image.open(self.image_files[index]).convert('RGB')

    target = cv2.imread(self.gd_files[index], -1)
    target[target == 255] = 1
    target = Image.fromarray(target)

    if self.transforms is not None:
      img = self.transforms(img)
    if self.target_transform is not None:
      target = (self.target_transform(target).squeeze() * 255).long()
    
    return img, target

  def __len__(self):
    return len(self.gd_files)
