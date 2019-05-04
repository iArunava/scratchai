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


def implemented(module, func):
  """
  Function to check if a function func exists in module.

  Arguments
  ---------
  module : any
           The module where the presence of function needs
           to be checked.
  func : str
         The name of the function in str whose presence 
         needs to be confirmed.

  Raises
  ------
  NotImplementedError : If the function is not present in module.
  """
  imp = getattr(module, func)
  if not callable(imp):
    raise NotImplementedError


def name_from_obj(obj):
  """
  This function returns the name of the object
  from its initialized instance.

  Arguments
  ---------
  obj : any
        The object whose name needs to be returned.

  Returns
  -------
  name : str
         The name of the object
  """
  # str(obj,__class__) -> "<class 'torch.optim.adam.Adam'>"
  cls_str = str(obj.__class__) if str(obj.__class__) != str(type) else str(obj)
  # [1:-2] -> "class 'torch.optim.adam.Adam"
  # .split('.')[-1] -> "Adam"
  return cls_str[1:-2].split('.')[-1].lower()


def load_pretrained(net:nn.Module, url:str, fname:str, nc:int=None):
  """
  Helps in Loading Pretrained networks
  """
  net.load_state_dict(load_from_pth(url, fname))
  # If nc != None, load a custom last linear layer
  # After freezing the rest of the network
  if nc is not None:
    for p in net.parameters():
      p.requires_grad_(False)
    net.fc = nn.Linear(512, nc)
  return net


def freeze(net:nn.Module):
  """
  Freeze the net.

  Arguments
  ---------
  net : nn.Module
        The net to freeze

  """
  for p in net.parameters():
    if p.requires_grad:
      p.requires_grad_(False)


def count_modules(net:nn.Module):
  """
  TODO
  """
  allm = []
  mdict = {}
  for m in net.modules():
    name = m.__class__.__name__
    if name in allm:
      mdict[name] += 1
    else:
      allm.append(name)
      mdict[name] = 1

  return mdict
