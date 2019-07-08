import random

from PIL import Image
from torchvision.transforms import functional as TF
from torchvision import transforms as T


# Move this function to imgutils
def pad_if_smaller(img, size, fill=0):
  out_width, out_height = img.size
  req_width, req_height = size
  if  out_width < req_width or out_height < req_height:
    pad_h = req_height - out_height if out_height < req_height else 0
    pad_w = req_width - out_width if out_width < req_width else 0
    img = TF.pad(img, (0, 0, pad_w, pad_h), fill=fill)
  return img


# NOTE No tests written for this
class Compose():
  def __init__(self,transforms):
    self.transforms = transforms

  def __call__(self, inputs, targets):
    for t in self.transforms:
      inputs, targets = t(inputs, targets)
    return inputs, targets


class RandomCrop():
  def __init__(self, size):
    print ('[INFO] Size is expected to be in format (width, height)')
    if isinstance(size, int): size = (size, size)
    else: assert isinstance(size, tuple) and len(size) == 2
    self.size = size

  def __call__(self, inputs, targets):
    inputs = pad_if_smaller(inputs, self.size)
    targets = pad_if_smaller(targets, self.size)
    crop_params_inputs = T.RandomCrop.get_params(inputs, self.size)
    crop_params_targets = T.RandomCrop.get_params(targets, self.size)
    inputs = TF.crop(inputs,   *crop_params_inputs)
    targets = TF.crop(targets, *crop_params_targets)
    return inputs, targets


class RandomHorizontalFlip():
  def __init__(self, flip_prob=0.5):
    self.flip_prob = flip_prob

  def __call__(self, inputs, targets):
    if random.random() < self.flip_prob:
      inputs  = TF.hflip(inputs)
      targets = TF.hflip(targets)
    return inputs, targets


# TODO Needs tests
class RandomResize():
  def __init__(self, min_size, max_size=None):
    self.min_size = min_size
    if max_size is None: max_size = min_size
    self.max_size = max_size

  def __call__(self, inputs, targets):
    size = random.randint(self.min_size, self.max_size)
    inputs = TF.resize(inputs, size)
    targets = TF.resize(targets, size, interpolation=Image.NEAREST)
    return inputs, targets


# TODO Needs tests
class ToTensor():
  def __call__(self, inputs, targets):
    inputs = TF.to_tensor(inputs)
    targets = torch.as_tensor(np.asarray(targets), dtype=torch.int64)
    return inputs, targets


# TODO Needs tests
class Normalize():
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, inputs, targets):
    inputs = TF.normalize(inputs, mean=self.mean, std=self.std)
    return inputs, targets
