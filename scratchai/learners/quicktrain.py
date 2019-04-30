"""
Wrappers to quickly train on common datasets.
"""

from scratchai.imgutils import get_trf
from scratchai.learners.clflearner import *
from scratchai._config import home

def train_mnist(net, **kwargs):
  """
  Train on MNIST with net.

  Arguments
  ---------
  net : nn.Module
        The net which to train.
  optim : nn.optim
          The optimizer to use.
  crit : nn.Module
         The criterion to use.
  lr : float
       The learning rate.
  wd : float
       The weight decay.
  bs : int
       The batch size.
  seed : int
         The particular seed to use.
  epochs : int
           The epcochs to train for.
  ckpt : str
         Path to the ckpt file.
  resume : bool
           If true, resume training from ckpt.
           else not.
  root : str
         The root where the datasets is or
         needs to be downloaded.

  """
  if 'optim' not in kwargs: kwargs['optim'] = optim.Adam
  if 'crit' not in kwargs: kwargs['crit'] = nn.CrossEntropyLoss
  if 'lr' not in kwargs: kwargs['lr'] = 3e-4
  if 'wd' not in kwargs: kwargs['wd'] = 0
  if 'bs' not in kwargs: kwargs['bs'] = 16
  if 'seed' not in kwargs: kwargs['seed'] = 123
  if 'epochs' not in kwargs: kwargs['epochs'] = 5
  if 'ckpt' not in kwargs: kwargs['ckpt'] = None
  if 'resume' not in kwargs: kwargs['resume'] = False
  if 'root' not in kwargs: kwargs['root'] = home
  
  if kwargs['resume']: assert kwargs['ckpt'] is not None

  for key, val in kwargs.items():
    print ('[INFO] Setting {} to {}.'.format(key, val))
  
  trf = get_trf('rr20_tt_normmnist')

  t = datasets.MNIST(kwargs['root'], train=True, download=True, transform=trf)
  v = datasets.MNIST(kwargs['root'], train=False, download=True, transform=trf)
  tloader = DataLoader(t, shuffle=True, batch_size=kwargs['bs'])
  vloader = DataLoader(v, shuffle=True, batch_size=kwargs['bs'])

  clf_fit(net, tloader, vloader, **kwargs)


def train_cifar10(net, **kwargs):
  """
  Train on MNIST with net.

  Arguments
  ---------
  net : nn.Module
        The net which to train.
  optim : nn.optim
          The optimizer to use.
  crit : nn.Module
         The criterion to use.
  lr : float
       The learning rate.
  wd : float
       The weight decay.
  bs : int
       The batch size.
  seed : int
         The particular seed to use.
  epochs : int
           The epcochs to train for.
  ckpt : str
         Path to the ckpt file.
  resume : bool
           If true, resume training from ckpt.
           else not.
  root : str
         The root where the datasets is or
         needs to be downloaded.

  """
  if 'optim' not in kwargs: kwargs['optim'] = optim.Adam
  if 'crit' not in kwargs: kwargs['crit'] = nn.CrossEntropyLoss
  if 'lr' not in kwargs: kwargs['lr'] = 3e-4
  if 'wd' not in kwargs: kwargs['wd'] = 0
  if 'bs' not in kwargs: kwargs['bs'] = 16
  if 'seed' not in kwargs: kwargs['seed'] = 123
  if 'epochs' not in kwargs: kwargs['epochs'] = 5
  if 'ckpt' not in kwargs: kwargs['ckpt'] = None
  if 'resume' not in kwargs: kwargs['resume'] = False
  if 'root' not in kwargs: kwargs['root'] = home
  
  if kwargs['resume']: assert kwargs['ckpt'] is not None

  for key, val in kwargs.items():
    print ('[INFO] Setting {} to {}.'.format(key, val))
  
  trf = get_trf('rz256_cc224_rr20_tt_normimgnet')

  t = datasets.CIFAR10(kwargs['root'], train=True, download=True, transform=trf)
  v = datasets.CIFAR10(kwargs['root'], train=False, download=True, transform=trf)
  tloader = DataLoader(t, shuffle=True, batch_size=kwargs['bs'])
  vloader = DataLoader(v, shuffle=True, batch_size=kwargs['bs'])

  clf_fit(net, tloader, vloader, **kwargs)
