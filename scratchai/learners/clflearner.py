import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scratchai.learners.metrics import accuracy

# This function is written lossely with respect to the library.
# This will be later integrated completely within the library.
# Lack of computational power to check things are working or not.

def clf_train(net, **kwargs):
  """
  This function is used to train the classification networks.
  """
  epochs = kwargs['epochs']
  lr = kwargs['lr']
  wd = kwargs['wd']
  mom = kwargs['mom']
  bs = kwargs['bs']
  best_acc = 0.

  trf = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
  
  # TODO Take the loaders as parameters to the function
  train = datasets.MNIST('./', train=True, download=True, transforms=trf)
  tloader = DataLoader(train, shuffle=True, batch_size=bs)
  val = datasets.MNIST('./', train=False, download=True, transforms=trf)
  vloader = DataLoader(val, shuffle=True, batch_size=bs)

  # TODO Take criterion as an option
  # TODO Take optimizer as an option
  crit = nn.CrossEntropyLoss()
  opti = optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=mom)

  for e in range(1, epochs+1):
    net.train()
    adjust_lr(opti, e)
    for ii, (data, labl) in enumerate(tqdm(tloader)):
      data, labl = data.to(device), labl.to(device)
      out = net(data)
      tloss = crit(out, labl)
      opti.zero_grad()
      tloss.backward()
      opti.step()
      acc = accuracy(out, labl)
      best_acc = acc if acc > best_acc else best_acc

    for ii, (data, labl) in enumerate(tqdm(vloader)):
      with torch.no_grad():
        data, labl = data.to(device), labl.to(device)
        out = net(data)
        vloss = crit(out, labl)
        acc = accuracy(out, labl)
      
    print ('Epoch: {}/{} .. Train Loss: {} .. Val Loss: {} .. Acc: {}'
           .format(e, epochs, tloss.item(), vloss.item(), acc)
    torch.save({'state_dict' : net.state_dict(), 'opti' : opti.state_dict()},
               'ckpt-{}-{}.pth'.format(e, acc)


def adjust_lr(opti, epoch, lr):
  """
  Sets learning rate to the initial LR decayed by 10 every 30 epochs.
  """
  lr = lr * (0.1 ** (epoch // 30))
  for pgroup in opti.param_groups:
    pgroup['lr'] = lr
