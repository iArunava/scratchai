import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scratchai.learners.metrics import accuracy
from scratchai.utils import AvgMeter


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
  return (a1mtr.avg, a5mtr.avg), vloss


def clf_train(net, tloader, opti:torch.optim, crit:nn.Module, **kwargs):
  # TODO Fix this
  if kwargs['topk'] != (1, 5):
    raise Exception('topk other than (1, 5) not supported for now.')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net.to(device)
  net.train()
  a1mtr = AvgMeter('train_acc1'); a5mtr = AvgMeter('train_acc5')
  tloss = 0
  try: crit = crit()
  except: pass
  for ii, (data, labl) in enumerate(tqdm(tloader)):
    data, labl = data.to(device), labl.to(device)
    out = net(data)
    loss = crit(out, labl)
    opti.zero_grad()
    loss.backward()
    opti.step()
    with torch.no_grad():
      tloss += loss.item()
      acc1, acc5 = accuracy(out, labl, topk=kwargs['topk'])
      a1mtr(acc1, data.size(0)); a5mtr(acc5, data.size(0))

  tloss /= len(tloader)
  return (a1mtr.avg, a5mtr.avg), tloss


def clf_fit(net:nn.Module, crit:nn.Module, opti:torch.optim, tloader, vloader, 
            **kwargs):
  """
  This function is used to train the classification networks.
  """
  epochs = kwargs['epochs']
  lr = kwargs['lr']
  lr_step = kwargs['lr_step']
  lr_decay = kwargs['lr_decay']
  seed = kwargs['seed'] if kwargs['seed'] else np.random.randint(100)

  bloss = float('inf')
  
  torch.manual_seed(seed)
  np.random.seed(seed)
  print ('[INFO] Setting torch seed to {}'.format(seed))

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  tlist = []
  vlist = []

  for e in range(1, epochs+1):
    if lr_step is not None and type(lr_step) == int and e%lr_step == 0:
      lr = adjust_lr(opti, lr, lr_decay)
    if lr_step is not None and type(lr_step) == list and e in lr_step:
      lr = adjust_lr(opti, lr, lr_decay)
      
    tacc, tloss = clf_train(net, tloader, opti, crit, topk=kwargs['topk'])
    vacc, vloss = clf_test(net, vloader, crit, topk=kwargs['topk'])
    
    tlist.append((tacc, tloss))
    vlist.append((vacc, vloss))

    if vloss < bloss:
      bloss = vloss
      torch.save({'net' : net.state_dict(), 'opti' : opti.state_dict()},
               'best_net-{}-{:.2f}.pth'.format(e, vacc[0]))
    
    # TODO The tloss and vloss needs a recheck.
    print ('Epoch: {}/{} - Train Loss: {:.3f} - Train Acc@1: {:.3f}' 
           '- Train Acc@5: {:.3f} - Val Loss: {:.3f} - Val Acc@1: {:.3f}'
           '- Val Acc@5: {:.3f}'.format(e, epochs, tloss, tacc[0], tacc[1], 
           vloss, vacc[0], vacc[1]))

    torch.save({'net' : net.cpu().state_dict(), 'opti' : opti.state_dict()},
               'net-{}-{:.2f}.pth'.format(e, vacc[0]))

  return tlist, vlist


def adjust_lr(opti, lr, lr_decay):
  """
  Sets learning rate to the initial LR decayed by 10 every `step` epochs.

  Arguments
  ---------
  opti : torch.optim
         The optimizer
  epoch : int
          The number of epochs completed.
  lr : float
       The initial learning rate.
  step : int
         The lr_step, at what interval lr needs to be updated.
  """
  # See: https://discuss.pytorch.org/t/adaptive-learning-rate/320/4
  lr /= (1. / lr_decay)
  for pgroup in opti.param_groups: pgroup['lr'] = lr
  print ('[INFO] Learning rate decreased to {}'.format(lr))
  return lr
