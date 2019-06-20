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

from scratchai.trainers.metrics import *
from scratchai.utils import AvgMeter


__all__ = ['Trainer', 'SegTrainer', 'SegAuxTrainer']


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
    
    self.t_lossmtr = AvgMeter('Train Loss')
    self.v_lossmtr = AvgMeter('Val Loss')

    # Temporary Variables (Variables that needs to be reinitialized after
    # each epoch
    self.loss  = 0.0

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
    
    for e in range(self.epochs):
      if self.to_adjust_lr: self.lr = adjust_lr(opti, lr, lr_decay)
      
      self.before_epoch_start()
      self.train()
      self.test()
      self.save_if_best(self.get_curr_val_loss)
      self.show_epoch_details(e)
      self.save_epoch_model(e)

      # Increase completed epochs by 1
      self.epochs_complete += 1
  
  def before_epoch_start(self):
    self.t_lossmtr.create_and_shift_to_new_slot()
    self.v_lossmtr.create_and_shift_to_new_slot()
    self.t1t_accmtr.create_and_shift_to_new_slot()
    self.t5t_accmtr.create_and_shift_to_new_slot()
    self.t1v_accmtr.create_and_shift_to_new_slot()
    self.t5v_accmtr.create_and_shift_to_new_slot()

  def save_if_best(self, metric):
    if metric() < self.best_loss:
      self.best_loss = metric()
      torch.save({'net'   : self.net.state_dict(), 
                  'optim' : self.optim.state_dict()},
                  'best_net-{:.2f}.pth'.format(metric()))

  def save_epoch_model(self, e):
    torch.save({'net' : self.net.cpu().state_dict(), 
                'opti' : self.optim.state_dict()},
                'net-{}-{:.2f}.pth'.format(e+1, self.get_curr_val_acc()))
    
  def get_curr_val_acc(self):
    return self.val_list[-1][0][0]
  
  def  get_curr_val_loss(self):
    return self.val_list[-1][1]

  def show_epoch_details(self, e):
    # TODO The tloss and vloss needs a recheck.
    print ('Epoch: {}/{} - Train Loss: {:.3f} - Train Acc@1: {:.3f}' 
           '- Train Acc@5: {:.3f} - Val Loss: {:.3f} - Val Acc@1: {:.3f}'
           '- Val Acc@5: {:.3f}'.format(e+1, self.epochs, self.train_list[-1][1],
                    self.train_list[-1][0][0], self.train_list[-1][0][1], 
                    self.get_curr_val_loss(), self.val_list[-1][0][0], 
                    self.val_list[-1][0][1]))


  def train(self):

    self.net.to(self.device)
    self.net.train()

    for ii, (data, labl) in enumerate(tqdm(self.train_loader)):
      data, labl = data.to(self.device), labl.to(self.device)
      out = self.net(data)
      
      self.get_loss(out, labl)
      self.update()
      self.update_metrics(out, labl, part='train')
    
    self.store_details(part='train')
  
  
  def store_details(self, part):
    if part == 'train':
      self.train_list.append(((self.t1t_accmtr.get_curr_slot_avg(), 
                               self.t5t_accmtr.get_curr_slot_avg()), 
                               self.t_lossmtr.get_curr_slot_avg()))
    elif part == 'val':
      self.val_list.append(((self.t1v_accmtr.get_curr_slot_avg(), 
                             self.t5v_accmtr.get_curr_slot_avg()),
                             self.v_lossmtr.get_curr_slot_avg()))
      
    
  def update_metrics(self, out, labl, part):
    with torch.no_grad():

      if part == 'train':
        self.t_lossmtr(self.loss.item(), self.batch_size)
        acc1, acc5 = accuracy(out, labl, topk=self.topk)
        self.t1t_accmtr(acc1)
        self.t5t_accmtr(acc5)

      elif part == 'val':
        self.v_lossmtr(self.loss.item(), self.batch_size)
        acc1, acc5 = accuracy(out, labl, topk=self.topk)
        self.t1v_accmtr(acc1)
        self.t5v_accmtr(acc5)

      else:
        raise ('Invalid Part! Not Supported!')
      
    
  def get_loss(self, out, target):
    self.loss = self.crit(out, target)
    
  def update(self):
    self.optim.zero_grad()
    self.loss.backward()
    self.optim.step()

  def test(self):
    self.net.to(self.device)
    self.net.eval()

    with torch.no_grad():
      for ii, (data, labl) in enumerate(tqdm(self.val_loader)):
        data, labl = data.to(self.device), labl.to(self.device)
        out = self.net(data)

        self.get_loss(out, labl)
        self.update_metrics(out, labl, part='val')
      
    self.store_details(part='val')
  
    
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
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.t_accmtr = AvgMeter('Train Pixel Accuracy')
    self.v_accmtr = AvgMeter('Val Pixel Accuracy')

    self.t_miumtr = AvgMeter('Train Mean IoU')
    self.v_miumtr = AvgMeter('Val Mean IoU')
  
  def before_epoch_start(self):
    self.t_lossmtr.create_and_shift_to_new_slot()
    self.v_lossmtr.create_and_shift_to_new_slot()
    self.t_accmtr.create_and_shift_to_new_slot()
    self.v_accmtr.create_and_shift_to_new_slot()
    self.t_miumtr.create_and_shift_to_new_slot()
    self.v_miumtr.create_and_shift_to_new_slot()
    
  def get_curr_val_acc(self):
    return self.val_list[-1][0]
  
  def get_curr_val_loss(self):
    return self.val_list[-1][2]

  def store_details(self, part):
    if part == 'train':
      self.train_list.append((self.t_accmtr.get_curr_slot_avg(), 
                              self.t_miumtr.get_curr_slot_avg(),
                              self.t_lossmtr.get_curr_slot_avg()))
    elif part == 'val':
      self.val_list.append((self.v_accmtr.get_curr_slot_avg(),
                            self.v_miumtr.get_curr_slot_avg(),
                            self.v_lossmtr.get_curr_slot_avg()))
    
  def show_epoch_details(self, e):
    # TODO The tloss and vloss needs a recheck.
    print ('Epoch: {}/{} - Train Loss: {:.3f} - Train Pixel Acc: {:.3f}' 
           '- Train Mean IoU: {:.3f} - Val Loss: {:.3f} - Val Pixel Acc: {:.3f}'
           '- Val Mean IoU: {:.3f}'.format(e+1, self.epochs, 
                                self.train_list[-1][2], self.train_list[-1][0], 
                                self.train_list[-1][1], self.val_list[-1][2], 
                                self.val_list[-1][0],   self.val_list[-1][1]))

  def update_metrics(self, out, labl, part):
    with torch.no_grad():
      nc = out.shape[1]
      out = torch.argmax(out, dim=1)

      out, labl = out.cpu().detach().numpy(), labl.cpu().detach().numpy()
      if part == 'train':
        self.t_lossmtr(self.loss.item(), self.batch_size)
        acc, per_class_acc = pixel_accuracy(nc, true=labl, pred=out)
        self.t_accmtr(acc)
        miu = mean_iu(nc, true=labl, pred=out)
        self.t_miumtr(miu)

      elif part == 'val':
        self.v_lossmtr(self.loss.item(), self.batch_size)
        acc, per_class_acc = pixel_accuracy(nc, true=labl, pred=out)
        self.v_accmtr(acc)
        miu = mean_iu(nc, true=labl, pred=out)
        self.v_miumtr(miu)

      else:
        raise ('Invalid Part! Not Supported!')

  def plot_train_vs_val(self):
    tacc = list(map(lambda x : x[0], self.train_list))
    tmiu = list(map(lambda x : x[1], self.train_list))
    tloss = list(map(lambda x : x[2], self.train_list))

    vacc = list(map(lambda x : x[0], self.val_list))
    vmiu = list(map(lambda x : x[1], self.val_list))
    vloss = list(map(lambda x : x[2], self.val_list))
    # TODO Don't use self.epochs!
    # If the fit function is called n times, then the number 
    # of epochs is n*self.epochs, which is captured by the below code
    # but not with self.epochs
    assert self.epochs_complete == len(self.train_list)
    epochs = np.arange(1, self.epochs_complete+1)
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, tacc, 'b--', label='Train Pixel Accuracy')
    plt.plot(epochs, vacc, 'b-', label='Val Pixel Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Pixel Accuracy')
    plt.title('Pixel Accuracy vs Epochs')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, tmiu, 'o--', label='Train Mean IoU')
    plt.plot(epochs, vmiu, 'o-', label='Val Mean IoU')
    plt.xlabel('Epochs'); plt.ylabel('Mean IoU')
    plt.title('Mean IoU vs Epochs')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, tloss, 'o--', label='Train Loss')
    plt.plot(epochs, vloss, 'o-', label='Val Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()

    plt.show()


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
    super().update_metrics(out['out'], labl, part)
