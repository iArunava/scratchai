"""
Wrappers to quickly train on common datasets.
"""

import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scratchai.imgutils import get_trf
from scratchai.trainers.trainer import *
from scratchai.trainers.gantrainer import GANTrainer
from scratchai.trainers.optimizer import Optimizer
from scratchai._config import home
from scratchai import utils
from scratchai._config import CIFAR10, MNIST, SKY_SEG
from scratchai.datasets import *



# =============================================================================
# 
# Classification Trainers
#
# =============================================================================

def dogs_and_cats(net, **kwargs):
  # NOTE Not Working
  """
  Train on MNIST with net.

  Arguments
  ---------
  net : nn.Module
        The net which to train.

  Returns
  -------
  trainer : scratchai.trainers.trainer
            The Trainer Object which holds the training details.
  """
  
  root, bs, opti, crit, evaluate, kwargs = \
          preprocess_opts(net, dset=MNIST, **kwargs)

  trf = get_trf('rz224_tt_normimgnet')

  t = DogsCats(root, image_set='train', download=True, transform=trf)
  v = DogsCats(root, image_set='val', download=True, transform=trf)
  tloader = DataLoader(t, shuffle=True, batch_size=bs)
  vloader = DataLoader(v, shuffle=True, batch_size=bs)

  
  dc_trainer = Trainer(net=net, criterion=crit, optimizer=opti, 
                         train_loader=tloader, val_loader=vloader, 
                         verbose=False, nc=10, **kwargs)
  dc_trainer.fit()
  dc_trainer.plot_train_vs_val()
  return dc_trainer


def mnist(net, **kwargs):
  """
  Train on MNIST with net.

  Arguments
  ---------
  net : nn.Module
        The net which to train.

  Returns
  -------
  trainer : scratchai.trainers.trainer
            The Trainer Object which holds the training details.
  """
  
  root, bs, opti, crit, evaluate, kwargs = \
          preprocess_opts(net, dset=MNIST, **kwargs)

  trf = get_trf('rr20_tt_normmnist')

  t = datasets.MNIST(root, train=True, download=True, transform=trf)
  v = datasets.MNIST(root, train=False, download=True, transform=trf)
  tloader = DataLoader(t, shuffle=True, batch_size=bs)
  vloader = DataLoader(v, shuffle=True, batch_size=bs)

  
  mnist_trainer = Trainer(net=net, criterion=crit, optimizer=opti, 
                         train_loader=tloader, val_loader=vloader, 
                         verbose=False, nc=10, **kwargs)
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
  
  Returns
  -------
  trainer : scratchai.trainers.trainer
            The Trainer Object which holds the training details.
  """
  root, bs, opti, crit, evaluate, kwargs = \
        preprocess_opts(net, dset=CIFAR10, **kwargs)

  trf = get_trf('pad4_rc32_tt_normimgnet')

  t = datasets.CIFAR10(root, train=True, download=True, transform=trf)
  v = datasets.CIFAR10(root, train=False, download=True, transform=trf)
  tloader = DataLoader(t, shuffle=True, batch_size=bs)
  vloader = DataLoader(v, shuffle=True, batch_size=bs) 
  cifar10_trainer = Trainer(net=net, criterion=crit, optimizer=opti, 
                            train_loader=tloader, val_loader=vloader, 
                            verbose=False, nc=10, **kwargs)
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
  
  Returns
  -------
  trainer : scratchai.trainers.trainer
            The Trainer Object which holds the training details.
  """
  opti, bs, crit, opti, evaluate, kwargs = preprocess_opts(net, **kwargs)

  #trf = get_trf('rz256_cc224_tt_normimgnet')
  trf = get_trf('rz32_tt_normimgnet')

  trainer = Trainer(net=net, criterion=crit, optimizer=opti, 
                    train_loader=tloader, val_loader=vloader, 
                    verbose=False, **kwargs)
  trainer.fit()
  trainer.plot_train_vs_val()
  return trainer


# =============================================================================
# 
# Segmentation QuickTrainers
#
# =============================================================================
def sky_segmentation(net, **kwargs):
  """
  Train on Sky Segmentation Dataset.

  Arguments
  ---------
  net : nn.Module
        The net which to train.

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
                      verbose=False, nc=2, **kwargs)
    sky.fit()
    sky.plot_train_vs_val()
  else:
    sky = SegEvaluater(net=net, criterion=crit, optimizer=opti, 
                      train_loader=tloader, val_loader=vloader, 
                      verbose=False, nc=2, **kwargs)
    sky.evaluate()

  return sky


# =============================================================================
# TODO I think "Trainers" is not the right word. 
# Gan Trainers
#
# =============================================================================

def dummy_gan(G, D, **kwargs):
  """
  Train on Dummy Data with GANs just to check everything is working.

  Arguments
  ---------

  """
  kwargs['gan'] = True
  kwargs['net2'] = D
  kwargs['bs'] = 2
  root, bs, opti, crit, evaluate, kwargs = preprocess_opts(G, **kwargs)
  trf = get_trf('tt_normimgnet')

  t = GANDummyData()
  tloader = DataLoader(t, shuffle=True, batch_size=bs)

  trainer = GANTrainer(G, net=D, criterion=crit, optimizer=opti,
                       train_loader=tloader, verbose=False, **kwargs)

  trainer.fit()
  return trainer


def mnist_gantrainer(G, D, **kwargs):
  """
  Train on Dummy Data with GANs just to check everything is working.

  Arguments
  ---------

  """
  kwargs['gan'] = True
  kwargs['net2'] = D
  root, bs, opti, crit, evaluate, kwargs = preprocess_opts(G, **kwargs)
  trf = get_trf('pad2_tt_normmnist')

  t = datasets.MNIST(root, train=True, download=True, transform=trf)
  #v = datasets.MNIST(root, train=False, download=True, transform=trf)
  tloader = DataLoader(t, shuffle=True, batch_size=bs)
  #vloader = DataLoader(v, shuffle=True, batch_size=bs)

  trainer = GANTrainer(G, net=D, criterion=crit, optimizer=opti,
                       train_loader=tloader, verbose=False, **kwargs)

  trainer.fit()
  return trainer


# =============================================================================
# 
# Preprocessing Operations
#
# =============================================================================

def preprocess_gan_opts(G, D, **kwargs):
  """
  Helper function that abstracts away the preprocessing of the 
  default kwargs for GANs.

  Arguments
  ---------
  """
  _kwargs = kwargs.copy()
  if 'optim' not in _kwargs: _kwargs['optim'] = optim.Adam
  if 'crit' not in _kwargs: _kwargs['crit'] = nn.BCELoss
  if 'lrD' not in _kwargs: _kwargs['lrD'] = 1e-3
  if 'lrG' not in _kwargs: _kwargs['lrG'] = 1e-3
  if 'betasD' not in _kwargs: _kwargs['betasD'] = (0.0, 0.99)
  if 'betasG' not in _kwargs: _kwargs['betasG'] = (0.0, 0.99)
  if 'wdD' not in _kwargs: _kwargs['wdD'] = 5e-4
  if 'wdG' not in _kwargs: _kwargs['wdG'] = 5e-4
  if 'momD' not in _kwargs: _kwargs['momD'] = 0.9
  if 'momG' not in _kwargs: _kwargs['momG'] = 0.9
  if 'nestvD' not in _kwargs: _kwargs['nestvD'] = False
  if 'nestvG' not in _kwargs: _kwargs['nestvG'] = False

  optis = (Optimizer(_kwargs['optim'], G, lr=_kwargs['lrG'], weight_decay=_kwargs['wdG'],
          momentum=_kwargs['momG'], nesterov=_kwargs['nestvG']), 
          Optimizer(_kwargs['optim'], D, lr=_kwargs['lrD'], weight_decay=_kwargs['wdD'],
          momentum=_kwargs['momD'], nesterov=_kwargs['nestvD']))
  
  crit = _kwargs['crit']()
  return crit, optis




def preprocess_opts(net, dset:str=None, **kwargs):
  """
  Helper function that abstracts away the preprocessing of the
  default kwargs as required for each dataset.

  Arguments
  ---------
  net : nn.Module
        The net to train. This is required cause the loading of ckpts happens
        in this function itself.
  dset : str, None
         Name of the dataset.
  kwargs : dict
           The dict with all the keys and values.
  Returns
  -------
  dict : kwargs
         The passed dict with all the values.

  Notes
  -----
  If gan, the net is the G, and I am passing an extra key in kwargs called net2
  which will contain the D. This won't be needed when the preprocesing is made
  into a class, so please change that while making this into a class.
  """
  # TODO Make the preprocess a Class Object for easy inheritance and usage
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
  if 'bias_2lr' not in kwargs: kwargs['bias_2lr'] = False
  if 'gan' not in kwargs: kwargs['gan'] = False
  
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
  bias_2lr = kwargs['bias_2lr']; kwargs.pop('bias_2lr', None)
  
  # Prepare the criterion and the optimizer
  if kwargs['gan']:
    # NOTE Please don't do this. what you did in the line below.
    # Remove it when preproces has its own object. Its not nice to remove the
    # crit all of a sudden like this.t 
    kwargs.pop('crit', None)
    # TODO Key net2 won't be needed after making these into a class.
    crit, opti = preprocess_gan_opts(net, kwargs['net2'], **kwargs)
    kwargs.pop('net2', None)
  else:
    opti = Optimizer(kwargs['optim'], net, lr=lr, weight_decay=wd, momentum=mom,
                     nesterov=nestv, bias_2lr=bias_2lr)
    crit = kwargs['crit']()
  """
  opti_name = utils.name_from_object(kwargs['optim'])
  train_params = [p for p in net.parameters() if p.requires_grad]
  if  opti_name == 'adam':
    opti = kwargs['optim'](train_params, lr=lr, weight_decay=wd)
  elif opti_name == 'sgd':
    opti = kwargs['optim'](train_params, lr=lr, nesterov=nestv,
                           weight_decay=wd, momentum=mom)
  else:
    raise NotImplementedError
  """

  # Resume from ckpt (if ckpt is not None)
  # TODO Handle the ckpt loading when its a gan
  if kwargs['gan'] is False:
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

