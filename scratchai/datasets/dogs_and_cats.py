import os
import cv2
import torch

from PIL import Image
from torchvision.datasets import VisionDataset
from glob import glob

from scratchai.utils import download_and_extract


class DogsCats(VisionDataset):
  """
  Loader for the Dogs and Cats Dataset.

  Arguments
  ---------
  root : str
         The path in which the dataset is downloaded or is present.

  image_set : str
              The image_set needed.
              Choices = ['train', val']

  download : bool
             To download or not, the dataset.

  transform : T.Compose
              The set of transforms needed for the images.

  Notes
  -----
  If image_set is set to 'train' then the first 10000 images are chosen
  from the dataset for both cats and dogs. And if the image_set is set to 'val'
  then all images from 10000 onwards gets selected for the validation data.
  """
  url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368'\
         '-6DEBA77B919F/kagglecatsanddogs_3367a.zip'

  def __init__(self, root='./', image_set='train', download=True, 
               transform=None):
    
    super().__init__(root, transform)
    
    self.root = os.path.abspath(root)
    self.transforms = transform
    
    if download: download_and_extract(self.url, self.root)
    
    cat_dir = os.path.join(self.root, 'PetImages/Cat/')
    dog_dir = os.path.join(self.root, 'PetImages/Dog/')
    
    if image_set == 'train':
      self.cat_files = glob(cat_dir + '/*')[:10000]
      self.dog_files = glob(dog_dir + '/*')[:10000]
    elif image_set == 'val':
      self.cat_files = glob(cat_dir + '/*')[10000:]
      self.dog_files = glob(dog_dir + '/*')[10000:]
    else:
      raise Exception('Unknown Image Set!')


  def __getitem__(self, index):
    cat_idx = True if index < len(self.cat_files) else False
    if not cat_idx: index = index - len(self.cat_files)

    img = Image.open(self.cat_files[index] if cat_idx \
          else self.dog_files[index]).convert('RGB')
    target = torch.Tensor([0 if cat_idx else 1])

    if self.transforms is not None:
      img = self.transforms(img)
    
    return img, target

  def __len__(self):
    return len(self.cat_files) + len(self.dog_files)
