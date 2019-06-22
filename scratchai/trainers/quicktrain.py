"""
Wrappers to quickly train on common datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scratchai.imgutils import get_trf
from scratchai.trainers.trainer import *
from scratchai._config import home
from scratchai import utils
from scratchai._config import CIFAR10, MNIST, SKY_SEG
from scratchai.datasets import *

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
  
  root, bs, opti, crit, kwargs = preprocess_opts(net, dset=MNIST, **kwargs)

  trf = get_trf('rr20_tt_normmnist')

  t = datasets.MNIST(root, train=True, download=True, transform=trf)
  v = datasets.MNIST(root, train=False, download=True, transform=trf)
  tloader = DataLoader(t, shuffle=True, batch_size=bs)
  vloader = DataLoader(v, shuffle=True, batch_size=bs)

  
  mnist_trainer = Trainer(net=net, criterion=crit, optimizer=opti, 
                         train_loader=tloader, val_loader=vloader, 
                         verbose=False, **kwargs)
  mnist_trainer.fit()
  mnist_trainer.plot_train_vs_val()
  return mnist_trainer


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
  opti, bs, crit, opti, kwargs = preprocess_opts(net, dset=CIFAR10, **kwargs)

  trf = get_trf('pad4_rc32_tt_normimgnet')

  t = datasets.CIFAR10(root, train=True, download=True, transform=trf)
  v = datasets.CIFAR10(root, train=False, download=True, transform=trf)
  tloader = DataLoader(t, shuffle=True, batch_size=bs)
  vloader = DataLoader(v, shuffle=True, batch_size=bs)
  
  cifar10_trainer = Trainer(net=net, criterion=crit, optimizer=opti, 
                            train_loader=tloader, val_loader=vloader, 
                            verbose=False, **kwargs)
  cifar10_trainer.fit()
  cifar10_trainer.plot_train_vs_val()
  return cifar10_trainer

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
  opti, bs, crit, opti, kwargs = preprocess_opts(net, **kwargs)

  #trf = get_trf('rz256_cc224_tt_normimgnet')
  trf = get_trf('rz32_tt_normimgnet')

  trainer = Trainer(net=net, criterion=crit, optimizer=opti, 
                    train_loader=tloader, val_loader=vloader, 
                    verbose=False, **kwargs)
  trainer.fit()
  trainer.plot_train_vs_val()
  return trainer


def sky_segmentation(net, **kwargs):
  """
  Train on Sky Segmentation Dataset.

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
  trainer : scratchai.learners.trainer.Trainer
            The Trainer Object which holds the training details.
  """
  root, bs, opti, crit, evaluate, kwargs = \
        preprocess_opts(net, dset=SKY_SEG, **kwargs)

  #trf = get_trf('pad4_rhf_cj.25_rc32_rr10_tt_normimgnet')
  trf_tr = get_trf('rz360_tt_normimgnet')
  trf_tt = get_trf('rz360.i2_tt_fm255')
  
  t = SkySegmentation(root, image_set='train', download=True, 
                              transform=trf_tr, target_transform=trf_tt)
  v = SkySegmentation(root, image_set='val', download=True, 
                              transform=trf_tr, target_transform=trf_tt)

  tloader = DataLoader(t, shuffle=True, batch_size=bs)
  vloader = DataLoader(v, shuffle=True, batch_size=bs)
  
  if not evaluate:
    sky = SegTrainer(net=net, criterion=crit, optimizer=opti, 
                      train_loader=tloader, val_loader=vloader, 
                      verbose=False, **kwargs)
    sky.fit()
    sky.plot_train_vs_val()
  else:
    sky = SegEvaluater(net=net, criterion=crit, optimizer=opti, 
                      train_loader=tloader, val_loader=vloader, 
                      verbose=False, **kwargs)
    sky.evaluate()

  return sky


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
  if 'evaluate' not in kwargs: kwargs['evaluate'] = False
  
  # Set Dataset specific values if dset is not None here

  for key, val in kwargs.items():
    print ('[INFO] Setting {} to {}.'.format(key, val))
  
  lr = kwargs['lr']; kwargs.pop('lr', None)
  wd = kwargs['wd']; kwargs.pop('wd', None)
  mom = kwargs['mom']; kwargs.pop('mom', None)
  nestv = kwargs['nestv']; kwargs.pop('nestv', None)
  bs = kwargs['bs']; kwargs.pop('bs', None)
  root = kwargs['root']; kwargs.pop('root', None)
  evaluate = kwargs['evaluate']; kwargs.pop('evaluate', None)

  crit = kwargs['crit']()
  opti_name = utils.name_from_object(kwargs['optim'])
  train_params = [p for p in net.parameters() if p.requires_grad]
  if  opti_name == 'adam':
    opti = kwargs['optim'](train_params, lr=lr, weight_decay=wd)
  elif opti_name == 'sgd':
    opti = kwargs['optim'](train_params, lr=lr, nesterov=nestv,
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
  kwargs.pop('crit', None); kwargs.pop('optim', None); kwargs.pop('ckpt', None)

  return root, bs, opti, crit, evaluate, kwargs

