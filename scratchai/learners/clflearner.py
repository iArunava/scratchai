import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scratchai.learners.metrics import accuracy

# This function is written lossely with respect to the library.
# This will be later integrated completely within the library.
# Lack of computational power to check things are working or not.

def test_net(net, vloader, crit:nn.Module=nn.CrossEntropyLoss):
  """
  Test the network.
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  vcorr = 0
  vloss = 0
  net.to(device)
  net.eval()
  crit = crit()
  with torch.no_grad():
    for ii, (data, labl) in enumerate(tqdm(vloader)):
      data, labl = data.to(device), labl.to(device)
      out = net(data)
      vloss += crit(out, labl).item()
      vcorr += (out.argmax(dim=1) == labl).float().sum()
  print ('\nAccuracy: {:.2f}%'.format(vcorr / (len(vloader)*vloader.batch_size)
                                                           * 100))


def clf_train(net, **kwargs):
  """
  This function is used to train the classification networks.
  """
  epochs = kwargs['epochs']
  lr = kwargs['lr']
  wd = kwargs['wd']
  bs = kwargs['bs']
  best_acc = 0.
  seed = kwargs['seed'] if kwargs['seed'] else np.random.randint(100)
  
  torch.manual_seed(seed)
  np.random.seed(seed)
  print ('[INFO] Setting torch seed to {}'.format(seed))

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  if kwargs['trf'] is None:
    trf = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])
  else:
    trf = kwargs['trf']
  
  # TODO Take the loaders as parameters to the function
  train = datasets.MNIST('./', train=True, download=True, transforms=trf)
  tloader = DataLoader(train, shuffle=True, batch_size=bs)
  val = datasets.MNIST('./', train=False, download=True, transforms=trf)
  vloader = DataLoader(val, shuffle=True, batch_size=bs)

  # TODO Take criterion as an option
  # TODO Take optimizer as an option
  crit = nn.CrossEntropyLoss()
  opti = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

  for e in range(1, epochs+1):
    net.train()
    tcorr = 0
    tloss = 0
    #adjust_lr(opti, e)
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

    vcorr = 0
    vloss = 0
    net.eval()
    with torch.no_grad():
      for ii, (data, labl) in enumerate(tqdm(vloader)):
        data, labl = data.to(device), labl.to(device)
        out = net(data)
        vloss += crit(out, labl).item()
        vcorr += (out.argmax(dim=1) == labl).float().sum()
    
    
    tloss /= len(tloader)
    vloss /= len(tloader)
    
    tacc = tcorr / (len(tloader)*bs)
    vacc = vcorr / (len(vloader)*bs)
    best_acc = vacc if vacc > best_acc else best_acc
    
    # TODO The tloss and vloss needs a recheck.
    print ('Epoch: {}/{} - Train Loss: {:.3f} - Val Loss: {:.3f} - '
           'Training Acc: {:.3f} - Val Acc: {:.3f}'
           .format(e, epochs, tloss, vloss, tacc, vacc))
    torch.save({'state_dict' : net.state_dict(), 'opti' : opti.state_dict()},
               'ckpt-{}-{}.pth'.format(e, vacc))


def train_mnist(net, **kwargs):
  """
  Train on MNIST with net.

  Arguments
  ---------
  net : nn.Module
        The net which to train.

  Returns
  -------

  """
  trf = transforms.Compose([transforms.RandomRotation(20),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))])
  clf_train(net, epochs=5, lr=3e-4, wd=1e-4, bs=16, seed=123, trf=trf)


def adjust_lr(opti, epoch, lr):
  """
  Sets learning rate to the initial LR decayed by 10 every 30 epochs.
  """
  lr = lr * (0.1 ** (epoch // 30))
  for pgroup in opti.param_groups:
    pgroup['lr'] = lr
