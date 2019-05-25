import torch
import torch.nn as nn
import torch.optim as optim

from scratchai.learners import metrics
from tabulate import tabulate
from tqdm import tqdm


__all__ = ['Learner', 'SegLearner']

  
class Learner():
  
  def __init__(self, net:nn.Module, loader=None, lr:float=1e-3, 
        epochs:int=10, mt:str='seg', sae:int=1, pe:int=1, wd:int=1e-4,
        trainiter:int=None, valiter:int=None, 
        crit:nn.Module=nn.CrossEntropyLoss, opt=optim.Adam, she:int=5,
        device='cuda', metrics:list=None):
    """
    A Learner Object. Base Class for all Learners.

    Arguments
    ---------
    net : nn.Module
       The model to train
    loader : TOFILL
        The DataLoader from where to get the data
    epochs : int
        The number of epochs
    she : int
       Interval in which to checkpoint
    pe : int
      Interval in which it prints the loss and other metrics
    wd : float
      Weight Decay for L2 Regularization

    """

    self.lr = lr
    self.epochs = epochs
    self.mt = mt
    self.device = device
    self.net = net.to(self.device)
    self.she = she
    self.sae = sae
    self.pe = pe
    self.loader = loader
    self.metrics = metrics
    self.wd = wd
    self.trainiter = trainiter
    self.valiter = valiter

    self.tlosses = []
    self.vlosses = []

    '''
    self.dltr = self.loader.get_dltr()
    self.dltt = self.loader.get_dltt()
    self.trainiter = len(self.dltr) // self.dltr.batch_size if not trainiter \
            else trainiter
    self.valiter = len(self.dltt) // self.dltt.batch_size if not valiter \
            else valiter
    '''
    self.crit = crit()
    self.opt = opt(net.parameters(), lr=self.lr, weight_decay=self.wd)

    self.h = 512
    self.w = 512
  
  def count_params(self):
      return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

  def fit(self):
    pass
  
  def calc_metrics(self):
    pass

  ################################################################################
  ######################## Ongoing Print Functions ###############################
  ################################################################################
  # TODO Use the torch-summary module and build on that.

  def conv_out_size(self, net):
    kh, kw = net.kernel_size if type(net.kernel_size) == tuple else (net.kernel_size, net.kernel_size)
    sh, sw = net.stride if type(net.stride) == tuple else (net.stride, net.stride)
    ph, pw = net.padding if type(net.padding) == tuple else (net.padding, net.padding)

    self.h = (int) ((self.h - kh + (2 * ph)) / sh) + 1
    self.w = (int) ((self.w - kw + (2 * pw)) / sw) + 1
    return self.h, self.w

  def unet_eblock_out(self, net):
    self.h = (int) ((self.h * 2) - 4)
    self.w = (int) ((self.w * 2) - 4)
    return self.h, self.w
  
  def summary(self):
    layers = [['Input'], [(self.h, self.w)]]
    print (tabulate(layers))
    self._summary(self.net)

  def _summary(self, net):
    layers = []
    for m in net.children():
      temp = []
      if isinstance(m, nn.Sequential):
        self._summary(m)

      elif isinstance(m, nn.Conv2d):
        temp.append('Conv2d({}, {}, {})'.format(m.kernel_size, m.stride, m.padding))
        temp.append('{}'.format(self.conv_out_size(m)))

      elif str(m.__class__).split('.')[-1][:-2] == 'UNet_EBlock':
        temp.append('UNet_EBlock({}, {})'.format(m.uc.in_channels, m.uc.out_channels))
        temp.append('{}'.format(self.unet_eblock_out(m)))

      elif str(m.__class__).split('.')[-1][:-2] == 'MaxPool2d':
        temp.append('MaxPool2d({}, {}, {})'.format(m.kernel_size, m.stride, m.padding))
        temp.append('{}'.format(self.conv_out_size(m)))
      else:
        temp.append('ReLU')
        temp.append('{}'.format((self.h, self.w)))
      
      if len(temp) > 0:
        layers.append(temp)
    
    print (tabulate(layers))


class SegLearner(Learner):
  """
  The Learner Object that helps train Segmentation Datasets.
  """
  
  # TODO Add pixel accuracy metric function to this
  avbl_metrics = ['miou']

  def __init__(self, *args, **kwargs):
    
    super().__init__(*args, **kwargs)
    
    '''
    self.dltr = self.loader.get_dltr()
    self.dltt = self.loader.get_dltt()
    '''
    self.trainiter = self.loader.len if not self.trainiter \
            else self.trainiter
    self.valiter = self.loader.len if not self.valiter \
            else self.valiter
    
    # Check if no extra metrics are requested than supported and 
    if self.metrics:
      assert len(SegLearner.avbl_metrics) >= len(self.metrics)
      assert all([True if metric in self.avbl_metrics else False \
                  for metric in self.metrics])

  def fit(self):
    """
    Method to help in training of Segmentation datasets
    """

    for e in range(1, self.epochs+1):
      trloss = 0
      self.net.train()

      for ii in tqdm(range(self.trainiter)):

        img, lab = next(iter(self.loader.get_batch()))

        # TODO hack to skip batches with size = 1
        if img.shape[0] == 1:
          continue

        img, lab = img.to(self.device), lab.to(self.device)
        # TODO Introduce transforms in get_batch
        #lab *= 255

        self.opt.zero_grad()
        out = self.net(img.float())
        loss = self.crit(out, lab.long())
        loss.backward()
        self.opt.step()

        trloss += loss.item()

        self.calc_metrics(out, lab)
        break

      self.tlosses.append(trloss)

      if ii % self.she == 0:
        #show_camvid(self.net, self.h, self.w)
        print ('Epoch {}/{}...'.format(e, self.epochs),
          'Loss {:3f}'.format(loss.item()))

      if e % self.sae == 0:
        checkpoint = {
          'epochs' : e,
          'model_state_dict' : self.net.state_dict(),
          'opt_state_dict' : self.opt.state_dict()
        }
        torch.save(checkpoint, './ckpt-{:2f}.pth'.format(trloss))
        print ('Model saved!')
        
      if e % self.pe == 0:
        print ('Epoch {}/{}...'.format(e, self.epochs),
            'Loss {:3f}'.format(trloss))

      if e % eval_every == 0 and self.dltt:
        with torch.no_grad():
          self.net.eval()

          eloss = 0

          for _ in tqdm(range(self.valiter)):
            img, lab = next(iter(self.get_batch()))

            img, lab = img.to(device), lab.to(device)
            out = self.net(img.float())
            loss = criterion(out, lab.long())
            eloss += loss.item()

          print ('Loss {:3f}'.format(eloss // self.valiter))

          self.vlosses.append(eloss)
      
      # TODO: An API where users can easily setup a scheduler
      #scheduler.step(train_loss)

    def calc_metrics(self, inp, lab):
      # TODO Implement
      return
      for metric in metrics:
        setattr(self, metric, getattr(metrics, metric)(inp, lab))
