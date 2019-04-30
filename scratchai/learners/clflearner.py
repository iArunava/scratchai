import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scratchai.learners.metrics import accuracy


def clf_test(net, vloader, crit:nn.Module=nn.CrossEntropyLoss):
  """
  This function helps in quickly testing the network.

  Arguments
  ---------
  net : nn.Module
        The net which to train.
  vloader : torch.nn.utils.DataLoader
            or a generator which returns the images and the labels
  
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  vcorr = 0
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
      vcorr += (out.argmax(dim=1) == labl).float().sum()
  
    vacc = accuracy(vcorr, len(vloader)*vloader.batch_size)
    vloss /= len(vloader)
  return vacc, vloss


def clf_train(net, tloader, opti, crit, **kwargs):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net.to(device)
  net.train()
  tcorr = 0
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
      tcorr += (out.argmax(dim=1) == labl).float().sum()

  tloss /= len(tloader)
  tacc = accuracy(tcorr, len(tloader)*tloader.batch_size)
  return tacc, tloss


def clf_fit(net, tloader, vloader, **kwargs):
  """
  This function is used to train the classification networks.
  """
  epochs = kwargs['epochs']
  lr = kwargs['lr']
  wd = kwargs['wd']
  seed = kwargs['seed'] if kwargs['seed'] else np.random.randint(100)

  best_acc = 0.
  
  torch.manual_seed(seed)
  np.random.seed(seed)
  print ('[INFO] Setting torch seed to {}'.format(seed))

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  crit = kwargs['crit']()
  opti = kwargs['optim'](net.parameters(), lr=lr, weight_decay=wd)
  
  # Resume from ckpt (if resume is True)
  if kwargs['resume']:
    ckpt = torch.load(kwargs['ckpt'])
    print ('[INFO] Looking for ket "opti_sd" in ckpt file...')
    opti.load_state_dict(ckpt['opti_sd'])
    print ('[INFO] Found and loaded the optimizer state_dict')
    print ('[INFO] Looking for ket "state_dict" in ckpt file...')
    net.load_state_dict(ckpt['state_dict'])
    print ('[INFO] Found and loaded the model state_dict')

  for e in range(1, epochs+1):
    tacc, tloss = clf_train(net, tloader, opti, crit)
    vacc, vloss = clf_test(net, vloader, crit)
    
    best_acc = vacc if vacc > best_acc else best_acc
    
    # TODO The tloss and vloss needs a recheck.
    print ('Epoch: {}/{} - Train Loss: {:.3f} - Training Acc: {:.3f}' 
           ' - Val Loss: {:.3f} - Val Acc: {:.3f}'
           .format(e, epochs, tloss, tacc, vloss, vacc))
    torch.save({'state_dict' : net.state_dict(), 'opti' : opti.state_dict()},
               'ckpt-{}-{}.pth'.format(e, vacc))


def adjust_lr(opti, epoch, lr):
  # TODO Needs testing
  """
  Sets learning rate to the initial LR decayed by 10 every 30 epochs.
  """
  lr = lr * (0.1 ** (epoch // 30))
  for pgroup in opti.param_groups:
    pgroup['lr'] = lr
