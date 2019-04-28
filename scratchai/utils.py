import torch
import torch.nn as nn
import os
from torchvision import transforms
from subprocess import call
from scratchai._config import home


def load_from_pth(url, fname='random', key='state_dict'):
  """
  Function to download/load the pth file and return the key mentioned.

  Arguments
  ---------
  url : str
        The url from which to download the file.
  key : str
        The key of the dict to return
  fname : str
          The name with which the file should be saved.
  Returns
  -------
  val : the type of element stored in the key (mostly torch.tensor)
        The value.
  """
  
  # TODO Download this file to a location where it doesn't need to 
  # be downloaded again and again.
  prefix = home
  if not os.path.isfile(prefix + fname + '.pth'):
    call(['wget', '-O', '{}{}.pth'.format(prefix, fname), url])
  ckpt = torch.load('{}{}.pth'.format(prefix, fname), map_location='cpu')
  return ckpt[key] if key in ckpt else ckpt


def get_trf(trfs:str):
  """
  A function to quickly get required transforms.

  Arguments
  ---------
  trfs : str
         An str that represents what transforms are needed. See Notes

  Returns
  -------
  trf : torch.transforms
        The transforms as a transforms object from torchvision.

  Notes
  -----
  >>> get_trf('rz256_cc224_tt_normimgnet')
  >>> transforms.Compose([transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transform.Normalize([0.485, 0.456, 0.406], 
                                              [0.229, 0.224, 0.225])])
  """
  # TODO Write tests
  # TODO Add more options
  trf_list = []
  for trf in trfs.split('_'):
    if trf.startswith('rz'):
      trf_list.append(transforms.Resize(int(trf[2:])))
    elif trf.startswith('cc'):
      trf_list.append(transforms.CenterCrop(int(trf[2:])))
    elif trf == 'tt':
      trf_list.append(transforms.ToTensor())
    elif trf == 'normimgnet':
      trf_list.append(transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225]))
    else:
      raise NotImplementedError

  return transforms.Compose(trf_list)
