"""
Wrappers to quickly train on common datasets.
"""

import numpy as np
import matplotlib.pyplot as plt

from scratchai.imgutils import get_trf
from scratchai.learners.clflearner import *
from scratchai._config import home
from scratchai import utils
from scratchai._config import CIFAR10, MNIST


def mnist(net, **kwargs):
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

  Returns
  -------
  tlist : list
          Contains list of n 2-tuples. where n == epochs
          and a tuple (a, b) where,
          a -> is the acc for the corresponding index
          b -> is the loss for the corresponding index
          for training
  vlist : list
          Contains list of n 2-tuples. where n == epochs
          and a tuple (a, b) where,
          a -> is the acc for the corresponding index
          b -> is the loss for the corresponding index
          for validation
  """
  
  opti, crit, kwargs = preprocess_opts(net, dset=MNIST, **kwargs)

  trf = get_trf('rr20_tt_normmnist')

  t = datasets.MNIST(kwargs['root'], train=True, download=True, transform=trf)
  v = datasets.MNIST(kwargs['root'], train=False, download=True, transform=trf)
  tloader = DataLoader(t, shuffle=True, batch_size=kwargs['bs'])
  vloader = DataLoader(v, shuffle=True, batch_size=kwargs['bs'])
  
  tlist, vlist = clf_fit(net, crit, opti, tloader, vloader, **kwargs)
  plt_tr_vs_tt(tlist, vlist)


def cifar10(net, **kwargs):
  """
  Train on CIFAR10 with net.

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
  ckpt : str, None
         Path to the ckpt file. If not None, training is started
         using this ckpt file. Defaults to None
  root : str
         The root where the datasets is or
         needs to be downloaded.
  
  Returns
  -------
  tlist : list
          Contains list of n 2-tuples. where n == epochs
          and a tuple (a, b) where,
          a -> is the acc for the corresponding index
          b -> is the loss for the corresponding index
          for training
  vlist : list
          Contains list of n 2-tuples. where n == epochs
          and a tuple (a, b) where,
          a -> is the acc for the corresponding index
          b -> is the loss for the corresponding index
          for validation
  """
  opti, crit, kwargs = preprocess_opts(net, dset=CIFAR10, **kwargs)

  trf = get_trf('pad4_rc32_tt_normimgnet')

  t = datasets.CIFAR10(kwargs['root'], train=True, download=True, transform=trf)
  v = datasets.CIFAR10(kwargs['root'], train=False, download=True, transform=trf)
  tloader = DataLoader(t, shuffle=True, batch_size=kwargs['bs'])
  vloader = DataLoader(v, shuffle=True, batch_size=kwargs['bs'])
  
  tlist, vlist = clf_fit(net, crit, opti, tloader, vloader, **kwargs)
  plt_tr_vs_tt(tlist, vlist)


def custom(net, tloader, vloader, **kwargs):
  """
  Train on a custom dataset with net.

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
  ckpt : str, None
         Path to the ckpt file. If not None, training is started
         using this ckpt file. Defaults to None
  root : str
         The root where the datasets is or
         needs to be downloaded.
  
  Returns
  -------
  tlist : list
          Contains list of n 2-tuples. where n == epochs
          and a tuple (a, b) where,
          a -> is the acc for the corresponding index
          b -> is the loss for the corresponding index
          for training
  vlist : list
          Contains list of n 2-tuples. where n == epochs
          and a tuple (a, b) where,
          a -> is the acc for the corresponding index
          b -> is the loss for the corresponding index
          for validation
  """
  opti, crit, kwargs = preprocess_opts(net, **kwargs)

  #trf = get_trf('rz256_cc224_tt_normimgnet')
  trf = get_trf('rz32_tt_normimgnet')

  tlist, vlist = clf_fit(net, crit, opti, tloader, vloader, **kwargs)
  plt_tr_vs_tt(tlist, vlist)


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
  if 'nestv' not in kwargs: kwargs['nestv'] = False
  if 'bs' not in kwargs: kwargs['bs'] = 16
  if 'seed' not in kwargs: kwargs['seed'] = 123
  if 'epochs' not in kwargs: kwargs['epochs'] = 5
  if 'topk' not in kwargs: kwargs['topk'] = (1, 5)
  if 'lr_step' not in kwargs: kwargs['lr_step'] = None
  if 'lr_decay' not in kwargs: kwargs['lr_decay'] = 0.2
  if 'ckpt' not in kwargs: kwargs['ckpt'] = None
  if 'root' not in kwargs: kwargs['root'] = home
  
  # Set Dataset specific values if dset is not None here

  for key, val in kwargs.items():
    print ('[INFO] Setting {} to {}.'.format(key, val))
  
  lr = kwargs['lr']
  wd = kwargs['wd']
  mom = kwargs['mom']

  crit = kwargs['crit']()
  opti_name = utils.name_from_object(kwargs['optim'])
  train_params = [p for p in net.parameters() if p.requires_grad]
  if  opti_name == 'adam':
    opti = kwargs['optim'](train_params, lr=lr, weight_decay=wd)
  elif opti_name == 'sgd':
    opti = kwargs['optim'](train_params, lr=lr, nesterov=kwargs['nestv'],
                           weight_decay=wd, momentum=mom)
  else:
    raise NotImplementedError

  # Resume from ckpt (if ckpt is not None)
  if kwargs['ckpt'] is not None:
    ckpt = torch.load(kwargs['ckpt'])
    print ('[INFO] Looking for key "opti" in ckpt file...')
    opti.load_state_dict(ckpt['opti'])
    # If optimizer is SGD, then momentum buffers needs to be moved to device
    # See: https://discuss.pytorch.org/t/runtimeerror-expected-type-torch-
    # floattensor-but-got-torch-cuda-floattensor-while-resuming-training/37936
    if opti_name == 'sgd':
      for p in opti.state.keys():
        buf = opti.state[p]['momentum_buffer']
        opti.state[p]['momentum_buffer'] = buf.cuda()
    print ('[INFO] Found and loaded the optimizer state_dict')
    print ('[INFO] Looking for key "net" in ckpt file...')
    net.load_state_dict(ckpt['net'])
    print ('[INFO] Found and loaded the model state_dict')

  # Pop keys from kwargs to avoid 
  # TypeError: got multiple values for 1 argument
  kwargs.pop('crit', None); kwargs.pop('opti', None); 

  return opti, crit, kwargs


def plt_tr_vs_tt(tlist, vlist):
  tacc = list(map(lambda x : x[0][0], tlist))
  tloss = list(map(lambda x : x[1], tlist))
  vacc = list(map(lambda x : x[0][0], vlist))
  vloss = list(map(lambda x : x[1], vlist))
  epochs = np.arange(1, len(tlist)+1)
  plt.plot(epochs, tacc, 'b--', label='Train Accuracy')
  plt.plot(epochs, vacc, 'b-', label='Val Accuracy')
  plt.plot(epochs, tloss, 'o--', label='Train Loss')
  plt.plot(epochs, vloss, 'o-', label='Val Loss')
  plt.xlabel('Epochs')
  plt.legend()
  plt.show()
