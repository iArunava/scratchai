"""
Base Class for Training Models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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
    self.topk         = topk
    self.crit         = criterion
    self.seed         = seed
    self.optim        = optimizer
    self.device       = torch.device(device)
    self.epochs       = epochs
    self.lr_step      = lr_step
    self.lr_decay     = lr_decay
    self.val_loader   = val_loader
    self.train_loader = train_loader
    self.batch_size   = self.train_loader.batch_size
    
    self.best_loss  = float('inf')
    self.epochs_complete = 0
    self.train_list = []
    self.val_list   = []
    self.to_adjust_lr = lr_step is not None and \
                        (type(lr_step) == int and e%lr_step == 0) or \
                        (type(lr_step) == list and e in lr_step)

    # Meters to keep track of Metrics
    self.t1t_accmtr = AvgMeter('Top 1 Train Accuracy')
    self.t5t_accmtr = AvgMeter('Top 5 Train Accuracy')

    self.t1v_accmtr = AvgMeter('Top 1 Val Accuracy')
    self.t5v_accmtr = AvgMeter('Top 5 Val Accuracy')
    
    # Temporary Variables (Variables that needs to be reinitialized after
    # each epoch
    self.loss  = 0.0
    self.tloss = 0.0
    self.vloss = 0.0

    if verbose: print (self.__str__())

  def fit(self):
    """
    This function is used to train the classification networks.
    """
    torch.manual_seed(self.seed)
    np.random.seed(self.seed)
    # TODO Make a function which prints out all the info.
    # And call that before calling fit, maybe in the constructor
    print ('[INFO] Setting torch seed to {}'.format(self.seed))
    
    for e in range(1, self.epochs+1):
      if self.to_adjust_lr: self.lr = adjust_lr(opti, lr, lr_decay)

      self.train()
      self.test()
      
      if self.vloss < self.best_loss:
        self.best_loss = self.vloss
        torch.save({'net'   : self.net.state_dict(), 
                    'optim' : self.optim.state_dict()},
                    'best_net-{:.2f}.pth'.format(self.val_list[-1][0][0]))
      
      # TODO The tloss and vloss needs a recheck.
      print ('Epoch: {}/{} - Train Loss: {:.3f} - Train Acc@1: {:.3f}' 
             '- Train Acc@5: {:.3f} - Val Loss: {:.3f} - Val Acc@1: {:.3f}'
             '- Val Acc@5: {:.3f}'.format(e, self.epochs, self.tloss, 
                      self.train_list[-1][0][0], self.train_list[-1][0][1], 
                      self.vloss, self.val_list[-1][0][0], 
                      self.val_list[-1][0][1]))

      torch.save({'net' : self.net.cpu().state_dict(), 
                  'opti' : self.optim.state_dict()},
                  'net-{}-{:.2f}.pth'.format(e, self.val_list[-1][0][0]))
      
      # Increase completed epochs by 1
      self.epochs_complete += 1

  def train(self):

    self.net.to(self.device)
    self.net.train()
    self.tloss = 0

    for ii, (data, labl) in enumerate(tqdm(self.train_loader)):
      data, labl = data.to(self.device), labl.to(self.device)
      out = self.net(data)
      
      self.get_loss(out, labl)
      self.update()
      self.update_metrics(out, labl, part='train')

    self.tloss /= len(self.train_loader)
    self.train_list.append(((self.t1t_accmtr.avg, self.t5t_accmtr.avg), 
                             self.tloss))
  

  def update_metrics(self, out, labl, part):
    with torch.no_grad():

      if part == 'train':
        self.tloss += self.loss.item()
        acc1, acc5 = accuracy(out, labl, topk=self.topk)
        self.t1t_accmtr(acc1, self.batch_size)
        self.t5t_accmtr(acc5, self.batch_size)

      elif part == 'val':
        self.vloss += self.loss.item()
        acc1, acc5 = accuracy(out, labl, topk=self.topk)
        self.t1v_accmtr(acc1, self.batch_size)
        self.t5v_accmtr(acc5, self.batch_size)

      else:
        raise ('Invalid Part! Not Supported!')
      
    
  def get_loss(self, out, target):
    self.loss = self.crit(out, target)
    
  def update(self):
    self.optim.zero_grad()
    self.loss.backward()
    self.optim.step()

  def test(self):
    self.vloss = 0
    self.net.to(self.device)
    self.net.eval()

    with torch.no_grad():
      for ii, (data, labl) in enumerate(tqdm(self.val_loader)):
        data, labl = data.to(self.device), labl.to(self.device)
        out = self.net(data)

        self.get_loss(out, labl)
        self.update_metrics(out, labl, part='val')
      
    self.vloss /= len(self.val_loader)
    self.val_list.append(((self.t1v_accmtr.avg, self.t5v_accmtr.avg),
                           self.vloss))
  
  def adjust_lr(self):
    """
    Sets learning rate to the initial LR decayed by 10 every `step` epochs.
    """
    # See: https://discuss.pytorch.org/t/adaptive-learning-rate/320/4
    self.lr /= (1. / self.lr_decay)
    for pgroup in self.optim.param_groups: pgroup['lr'] = self.lr
    print ('[INFO] Learning rate decreased to {}'.format(lr))

  def plot_train_vs_val(self):
    tacc = list(map(lambda x : x[0][0], self.train_list))
    tloss = list(map(lambda x : x[1], self.train_list))
    vacc = list(map(lambda x : x[0][0], self.val_list))
    vloss = list(map(lambda x : x[1], self.val_list))
    # TODO Don't use self.epochs!
    # If the fit function is called n times, then the number 
    # of epochs is n*self.epochs, which is captured by the below code
    # but not with self.epochs
    assert self.epochs_complete == len(self.train_list)
    epochs = np.arange(1, self.epochs_complete+1)
    
    plt.plot(epochs, tacc, 'b--', label='Train Accuracy')
    plt.plot(epochs, vacc, 'b-', label='Val Accuracy')
    plt.plot(epochs, tloss, 'o--', label='Train Loss')
    plt.plot(epochs, vloss, 'o-', label='Val Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

  def __str__(self):
    """
    Shows the state of this trainer object
    """
    s = ""
    for var in vars(self):
      if var in ['net', 'train_loader', 'val_loader']: continue
      s += ('{} set to {}\n'.format(var, getattr(self, var)))
    return s



class SegTrainer(Trainer):
  """
  Trainer Object to train a Segmentation Model.
  """
  def update_metrics(self, out, labl, part):
    with torch.no_grad():
      if part == 'train':
        self.tloss += self.loss.item()
      elif part == 'val':
        self.vloss += self.loss.item()
      else:
        raise ('Invalid Part! Not Supported!')



class SegAuxTrainer(SegTrainer):
  """
  Trainer Object to train a Auxiliary Model.

  Arguments
  ---------
  loss_wdict : dict
               A dict holding name, value pairs where name should
               be the same as the names in the dict that is retured
               from the model and the values indicates how much weight
               the loss for that should be assigned to.
  """
  def __init__(self, loss_wdict:dict=None, **kwargs):
    super().__init__(**kwargs)

    if loss_wdict is None: loss_wdict = {'out': 1.0, 'aux': 0.5}
    self.loss_wdict = loss_wdict

  def get_loss(self, out, target):
    self.loss = 0
    for name, x in out.items():
      weight = self.loss_wdict[name] if name in self.loss_wdict else 0.5
      self.loss += weight * self.crit(x, target)

  def update_metrics(self, out, labl, part):
    with torch.no_grad():
      out = out['out']
      if part == 'train':
        self.tloss += self.loss.item()
      elif part == 'val':
        self.vloss += self.loss.item()
      else:
        raise ('Invalid Part! Not Supported!')
