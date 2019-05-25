import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torchvision import transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
from scratchai.utils import freeze, Topk
from scratchai.learners.metrics import accuracy
from scratchai.attacks.attacks import *
from scratchai.imgutils import get_trf
from scratchai.utils import name_from_object
from scratchai._config import CIFAR10, MNIST, IMGNET12


def optimize_linear(grads, eps, ordr):
  """
  Solves for optimal input to a linear function under a norm constraint.

  Arguments
  ---------
  grads : torch.Tensor
           The gradients of the input.
  eps : float
        Scalar specifying the constraint region.
  ordr : [np.inf, 1, 2] 
         Order of norm constraint.

  Returns
  -------
  opt_pert : torch.Tensor
             Optimal Perturbation.
  """

  red_ind = list(range(1, len(grads.size())))
  azdiv = torch.tensor(1e-12, dtype=grads.dtype, device=grads.device)

  if ordr == np.inf:
    opt_pert = torch.sign(grads)

  elif ordr == 1:
    abs_grad = torch.abs(grads)
    sign = torch.sign(grads)
    ori_shape = [1] * len(grads.size())
    ori_shape[0] = grads.size(0)

    max_abs_grad, _ = torch.max(abs_grad.view(grads.size(0), -1), 1)
    max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).float()
    num_ties = max_mask
    for red_scalar in red_ind:
      num_ties = torch.sum(num_ties, red_scalar, keepdims=True)
    opt_pert = sign * max_mask / num_ties
    # TODO tests

  elif ordr == 2:
    # TODO
    square = torch.max(azdiv, torch.sum(grads ** 2, red_ind, keepdim=True))
    opt_pert = grads / torch.sqrt(square)
    # TODO tests
  else:
    raise NotImplementedError('Only L-inf, L1 and L2 norms are '
                              'currently implemented.')

  scaled_pert = eps * opt_pert
  return scaled_pert


def clip_eta(eta, ord, eps):
  """
  Helper fucntion to clip the perturbation to epsilon norm ball.

  Args:
      eta: A tensor with the current perturbation
      ord: Order of the norm (mimics Numpy)
           Possible values: np.inf, 1 or 2.
      eps: Epsilon, bound of the perturbation.
  """

  # Clipping perturbation eta to self.ord norm ball
  if ord not in [np.inf, 1, 2]:
    raise ValueError('ord must be np.inf, 1, or 2.')

  reduce_ind = list(range(1, len(eta.shape)))
  azdiv = torch.tensor(1e-12)

  if ord == np.inf:
    eta = torch.clamp(eta, -eps, eps)
  else:
    if ord == 1:
      raise NotImplementedError("The expression below is not the correct way"
                                    " to project onto the L1 norm ball.")
      norm = torch.max(azdiv, torch.mean(torch.abs(eta), reduce_ind))

    elif ord == 2:
      # azdiv(avoid_zero_div) must go inside sqrt to avoid a divide by zero
      # in the gradient through this operation.
      norm = torch.sqrt(torch.max(azdiv, torch.mean(eta**2, reduce_ind)))

    # We must clip to within the norm ball, not 'normalize' onto the
    # surface of the ball
    factor = min(1., eps / norm)
    eta *= factor

  return eta



##################################################################
######### Functions to help benchmark attacks ####################
##################################################################

def benchmark_atk(atk, net:nn.Module, **kwargs):
  """
  Helper function to benchmark using a particular attack
  on a particular dataset. All benchmarks that are present in this
  repository are created using this function.

  Arguments
  ---------
  atk : scratchai.attacks.attacks
        The attack on which to use.
  net : nn.Module
        The net which is to be attacked.
  root : str
         The root directory of the dataset.
  dfunc : function
          The function that can take the root and torchvision.transforms
          and return a torchvision.Datasets object
          Defaults to datasets.ImageFolder
  trf : torchvision.Transforms
        The transforms that you want to apply.
        Defaults to (get_trf('rz256_cc224_tt_normimgnet')
  bs : int
       The batch size. Defaults to 4.
  
  """
  
  loader, topk, kwargs = pre_benchmark_atk(**kwargs)

  freeze(net)
  print ('[INFO] Net Frozen!')
  atk = atk(net, **kwargs)
  atk_name = name_from_object(atk)
  net_name = name_from_object(net)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  loss = 0; adv_loss = 0
  oatopk = Topk('original accuracy', topk)
  aatopk = Topk('adversarial accuracy', topk)
  net.to(device); net.eval()
  crit = nn.CrossEntropyLoss()

  for ii, (data, labl) in enumerate(tqdm(loader)):
    adv_data, data = atk(data.to(device).clone()), data.to(device)
    labl = labl.to(device)
    adv_out = net(adv_data); out = net(data)
    loss += crit(out, labl).item()
    adv_loss += crit(adv_out, labl).item()
    acc = accuracy(out, labl, topk)
    adv_acc = accuracy(adv_out, labl, topk)
    oatopk.update(acc, data.size(0)); aatopk.update(adv_acc, data.size(0))
  loss /= len(loader)
  adv_loss /= len(loader)
  
  print ('\nAttack Summary on {} with {} attack:'.format(net_name, atk_name))
  print ('-'*45)
  print (oatopk)
  print ('-'*35)
  print (aatopk)


def pre_benchmark_atk(**kwargs):
  """
  Helper function that sets all the defaults while performing checks
  for all the options passed before benchmarking attacks.
  """

  # Set the Default options if nothing explicit provided
  def_dict = {
    'bs'       : 4,
    'trf'      : get_trf('rz256_cc224_tt_normimgnet'),
    'dset'     : 'NA',
    'root'     : './',
    'topk'     : (1, 5),
    'dfunc'    : datasets.ImageFolder,
    'download' : True,
  }

  for key, val in def_dict.items(): 
    if key not in kwargs: kwargs[key] = val
  
  if kwargs['dset'] == 'NA':
    if 'loader' not in kwargs:
      dset = kwargs['dfunc'](kwargs['root'], transform=kwargs['trf'])
      loader = DataLoader(dset, batch_size=kwargs['bs'], num_workers=2)
    else:
      loader = kwargs['loader']
  
  # Set dataset specific functions here
  else:
    if kwargs['dset'] == IMGNET12:
      dset = datasets.ImageNet(kwargs['root'], split='test',
                      download=kwargs['download'], transform=kwargs['trf'])
    elif kwargs['dset'] == MNIST:
      kwargs['trf'] = get_trf('tt_normmnist')
      kwargs['dfunc'] = datasets.MNIST
      dset = kwargs['dfunc'](kwargs['root'], train=False, 
                     download=kwargs['download'], transform=kwargs['trf'])
    else: raise

    loader = DataLoader(dset, shuffle=False, batch_size=kwargs['bs'])
  topk = kwargs['topk']
    
  for key, val in kwargs.items():
    print ('[INFO] Setting {} to {}.'.format(key, kwargs[key]))

  # Deleting keys that is used just for benchmark_atk() function is 
  # important as the same kwargs dict is passed to initialize the attack
  # So, otherwise the attack will throw an exception
  for key in def_dict:
    del kwargs[key]
  if 'loader' in kwargs: del kwargs['loader']

  return loader, topk, kwargs
