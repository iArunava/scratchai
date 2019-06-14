"""
Base Class for Training Models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scratchai.learners.metrics import accuracy
from scratchai.utils import AvgMeter


class Trainer():
  """
  Arguments
  ---------
  lr           : float
                 Learning rate.
  epochs       : int
                 The number of epochs to train.
  train_loader : nn.utils.DataLoader
                 The TrainLoader
  val_loader   : nn.utils.DataLoader
                 The Validation Loader.
  """
  def __init__(self, net, train_loader, val_loader, lr=1e-3, epochs=5, 
               criterion=nn.CrossEntropyLoss, optimizer=optim.Adam, 
               lr_step=None, lr_decay=0.2, seed=123, device='cuda', 
               topk:tuple=(1, 5), verbose:bool=True):
    
    # TODO Fix this
    if topk != (1, 5):
      raise Exception('topk other than (1, 5) not supported for now.')

    if device == 'cuda' and not torch.cuda.is_available():
      device = 'cpu'
      print ('[INFO] Changing device to CPU as no GPU is available!')

    self.lr           = lr
    self.net          = net
    self.crit         = criterion
    self.seed         = seed
    self.optim        = optimizer
    self.device       = torch.device(device)
    self.epochs       = epochs
    self.lr_step      = lr_step
    self.lr_decay     = lr_decay
    self.val_loader   = val_loader
    self.train_loader = train_loader
    
    self.best_loss  = float('inf')
    self.train_list = []
    self.val_list   = []
    self.to_adjust_lr = lr_step is not None and \
                        (type(lr_step) == int and e%lr_step == 0) or \ 
                        (type(lr_step) == list and e in lr_step)

    if verbose: show_state()

  def fit():
    """
    This function is used to train the classification networks.
    """
    torch.manual_seed(self.seed)
    np.random.seed(self.seed)
    # TODO Make a function which prints out all the info.
    # And call that before calling fit, maybe in the constructor
    print ('[INFO] Setting torch seed to {}'.format(seed))

    for e in range(1, self.epochs+1):
      if to_adjust_lr: self.lr = adjust_lr(opti, lr, lr_decay)

      self.train()
      vacc, vloss = self.test()
      
      self.val_list.append((vacc, vloss))

      if vloss < self.best_loss:
        self.best_loss = vloss
        torch.save({'net'   : net.state_dict(), 
                    'optim' : self.optim.state_dict()},
                    'best_net-{:.2f}.pth'.format(vacc[0]))
      
      # TODO The tloss and vloss needs a recheck.
      print ('Epoch: {}/{} - Train Loss: {:.3f} - Train Acc@1: {:.3f}' 
             '- Train Acc@5: {:.3f} - Val Loss: {:.3f} - Val Acc@1: {:.3f}'
             '- Val Acc@5: {:.3f}'.format(e, epochs, tloss, tacc[0], tacc[1], 
             vloss, vacc[0], vacc[1]))

      torch.save({'net' : net.cpu().state_dict(), 'opti' : opti.state_dict()},
                 'net-{}-{:.2f}.pth'.format(e, vacc[0]))

  def train(self):

    net.to(self.device)
    net.train()
    a1mtr = AvgMeter('train_acc1'); a5mtr = AvgMeter('train_acc5')
    tloss = 0
    try: crit = crit()
    except: pass
    for ii, (data, labl) in enumerate(tqdm(self.train_loader)):
      data, labl = data.to(self.device), labl.to(self.device)
      out = net(data)
      loss = self.crit(out, labl)
      self.optim.zero_grad()
      loss.backward()
      self.optim.step()
      with torch.no_grad():
        tloss += loss.item()
        acc1, acc5 = accuracy(out, labl, topk=self.topk)
        a1mtr(acc1, data.size(0)); a5mtr(acc5, data.size(0))

    tloss /= len(self.train_loader)
    self.train_list.append((a1mtr.avg, a5mtr.avg), tloss)

  def adjust_lr(self):
    """
    Sets learning rate to the initial LR decayed by 10 every `step` epochs.
    """
    # See: https://discuss.pytorch.org/t/adaptive-learning-rate/320/4
    self.lr /= (1. / self.lr_decay)
    for pgroup in self.optim.param_groups: pgroup['lr'] = self.lr
    print ('[INFO] Learning rate decreased to {}'.format(lr))

  def __str__(self):
    """
    Shows the state of this trainer object
    """
    s = ""
    for key, value in vars(self):
      print ('{} set to {}'.format(key, value))

def clf_test(net, vloader, crit:nn.Module=nn.CrossEntropyLoss, topk=(1,5)):
  """
  This function helps in quickly testing the network.

  Arguments
  ---------
  net : nn.Module
        The net which to train.
  vloader : torch.nn.utils.DataLoader
            or a generator which returns the images and the labels
  
  """
  # TODO Fix this
  if topk != (1, 5):
    raise Exception('topk other than (1, 5) not supported for now.')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  a1mtr = AvgMeter('test_acc1'); a5mtr = AvgMeter('test_acc5') #
  vloss = 0
  net.to(device)
  net.eval()
  try: crit = crit()
  except: pass
  with torch.no_grad():
    for ii, (data, labl) in enumerate(tqdm(vloader)):
      data, labl = data.to(device), labl.to(device)
      out = net(data)
      vloss += crit(out, labl).item()
      acc1, acc5 = accuracy(out, labl, topk=topk)
      a1mtr(acc1, data.size(0)); a5mtr(acc5, data.size(0))
    
    vloss /= len(vloader)

  self.val_list.append((a1mtr.avg, a5mtr.avg), vloss)


