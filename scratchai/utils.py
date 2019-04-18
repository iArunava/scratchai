import torch
import torch.nn as nn
from subprocess import call

def load_from_pth(url, key='state_dict'):
  """
  Function to download/load the pth file and return the key mentioned.

  Arguments
  ---------
  url : str
        The url from which to download the file.
  key : str
        The key of the dict to return

  Returns
  -------
  val : the type of element stored in the key (mostly torch.tensor)
        The value.
  """
  
  # TODO Download this file to a location where it doesn't need to 
  # be downloaded again and again.
  call(['wget', '-O', '/tmp/random.pth', url])
  ckpt = torch.load('/tmp/random.pth')
  return ckpt[key]
