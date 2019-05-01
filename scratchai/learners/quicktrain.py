"""
Wrappers to quickly train on common datasets.
"""

from scratchai.imgutils import get_trf
from scratchai.learners.clflearner import *
from scratchai._config import home
from scratchai import utils

MNIST   = 'mnist'
CIFAR10 = 'cifar10'


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
  
  opti, crit, kwargs = preprocess_opts(net, dset=MNIST, **kwargs)

  trf = get_trf('rr20_tt_normmnist')

  t = datasets.MNIST(kwargs['root'], train=True, download=True, transform=trf)
  v = datasets.MNIST(kwargs['root'], train=False, download=True, transform=trf)
  tloader = DataLoader(t, shuffle=True, batch_size=kwargs['bs'])
  vloader = DataLoader(v, shuffle=True, batch_size=kwargs['bs'])
  
  tlist, vlist = clf_fit(net, crit, opti, tloader, vloader, **kwargs)


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
  opti, crit, kwargs = preprocess_opts(net, dset=CIFAR10, **kwargs)

  trf = get_trf('rr2_tt_normimgnet')

  t = datasets.CIFAR10(kwargs['root'], train=True, download=True, transform=trf)
  v = datasets.CIFAR10(kwargs['root'], train=False, download=True, transform=trf)
  tloader = DataLoader(t, shuffle=True, batch_size=kwargs['bs'])
  vloader = DataLoader(v, shuffle=True, batch_size=kwargs['bs'])
  
  tlist, vlist = clf_fit(net, a, crit, opti, tloader, vloader, **kwargs)


def preprocess_opts(net, dset:str=None, **kwargs):
  """
  Helper function that abstracts away the preprocessing of the
  default kwargs as required for each dataset.

  Arguments
  ---------
  net : nn.Module
        The net to train.
  dset : str, None
         Name of the dataset.
  kwargs : dict
           The dict with all the keys and values.
  Returns
  -------
  dict : kwargs
         The passed dict with all the values.
  """

  if 'optim' not in kwargs: kwargs['optim'] = optim.SGD
  if 'crit' not in kwargs: kwargs['crit'] = nn.CrossEntropyLoss
  if 'lr' not in kwargs: kwargs['lr'] = 3e-4
  if 'wd' not in kwargs: kwargs['wd'] = 0
  if 'mom' not in kwargs: kwargs['mom'] = 0.9
  if 'bs' not in kwargs: kwargs['bs'] = 16
  if 'seed' not in kwargs: kwargs['seed'] = 123
  if 'epochs' not in kwargs: kwargs['epochs'] = 5
  if 'ckpt' not in kwargs: kwargs['ckpt'] = None
  if 'resume' not in kwargs: kwargs['resume'] = False
  if 'root' not in kwargs: kwargs['root'] = home
  
  # Setting Dataset specific values if dset is not None
  if dset == CIFAR10:
    if 'optim' not in kwargs: kwargs['optim'] = optim.SGD
  elif dset == MNIST:
    if 'optim' not in kwargs: kwargs['optim'] = optim.Adam

  if kwargs['resume']: assert kwargs['ckpt'] is not None
  
  for key, val in kwargs.items():
    print ('[INFO] Setting {} to {}.'.format(key, val))
  
  lr = kwargs['lr']
  wd = kwargs['wd']
  mom = kwargs['mom']

  crit = kwargs['crit']()
  opti_name = utils.name_from_object(kwargs['optim'])
  if  opti_name == 'adam':
    opti = kwargs['optim'](net.parameters(), lr=lr, weight_decay=wd)
  elif opti_name == 'sgd':
    opti = kwargs['optim'](net.parameters(), lr=lr, 
                           weight_decay=wd, momentum=mom)
  else:
    raise NotImplementedError

  # Resume from ckpt (if resume is True)
  if kwargs['resume']:
    ckpt = torch.load(kwargs['ckpt'])
    print ('[INFO] Looking for key "opti" in ckpt file...')
    opti.load_state_dict(ckpt['opti'])
    print ('[INFO] Found and loaded the optimizer state_dict')
    print ('[INFO] Looking for key "net" in ckpt file...')
    net.load_state_dict(ckpt['net'])
    print ('[INFO] Found and loaded the model state_dict')

  # Pop keys from kwargs to avoid 
  # TypeError: got multiple values for 1 argument
  kwargs.pop('crit', None); kwargs.pop('opti', None); 

  return opti, crit, kwargs
