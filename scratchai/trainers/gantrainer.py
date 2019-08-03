"""
This file stores the code that allows to train GANs.
"""

import torch
import torch.nn as nn

from tqdm import tqdm

from scratchai.imgutils import imshow
from scratchai.trainers.trainer import Trainer
from scratchai.utils import AvgMeter


__all__ = ['GANTrainer']


class GANTrainer(Trainer):
  """
  Class to Train GANs Normally.

  Notes
  -----
  This class assumes that the Discriminator (D) is passed in as the first 
  argument and all other arguments specially the `net` argument is the 
  Generator object.
  """
  def __init__(self, G, **kwargs):
    super().__init__(**kwargs)
    self.G = G
    self.D = self.net

    self.real = 1
    self.fake = 0
    # TODO Please check if the label can be created and used like this
    self.label = torch.full((self.batch_size,), self.real, device=self.device)
    
    # Setting up the optimizers
    assert isinstance(self.optimizer, tuple) and len(self.optimizer) == 2
    print ('[INFO] The first optimizer is assumed to be for the Generator!')
    self.optG, self.optD = self.optimizer
    
    self.lossD = AvgMeter('Discriminator Loss')
    self.lossG = AvgMeter('Generator Loss')

    self.train_list = []

    self.z_size = 100
  

  def before_epoch_start(self):
    self.lossD.create_and_shift_to_new_slot()
    self.lossG.create_and_shift_to_new_slot()


  def fit_body(self, e):
    self.before_epoch_start()
    self.train()
    self.show_epoch_details(e)
    self.save_epoch_model(e)

    
  def create_random_noise(self, bs=None):
    # The most simple way to create the noise
    bs = self.batch_size if bs is None else bs
    return torch.randn(bs, self.z_size, 1, 1, device=self.device)


  def trainD(self, data):
    real = self.D(data)
    self.label.fill_(self.real)
    rloss = self.criterion(real, self.label)

    self.optD.zero_grad()
    
    # TODO Check whether this can be made to pass backward at once with rloss
    # at the end, that is keep both the backwards together with opt.step
    # if so, abstract the whole thing into the updateD() function
    rloss.backward()
    
    # Create random noise to be passed to the generator
    z = self.create_random_noise()
    
    fimages = self.G(z)
    self.label.fill_(self.fake)
    # NOTE This detach is very much needed for things to work!
    # Its the first Big Bug I faced while coding DCGAN
    fake = self.D(fimages.detach())
    floss = self.criterion(fake, self.label)
    floss.backward()

    self.dloss = rloss + floss
    self.optD.step()

    return fimages
    
    
  def trainG(self, data):
    self.label.fill_(self.real)
    fake = self.D(data)
    self.gloss = self.criterion(fake, self.label)
    # TODO Move the below two lines in updateG()
    self.gloss.backward()
    self.optG.step()
   
  
  def train_body(self):
    for ii, (rimages, _) in enumerate(tqdm(self.train_loader)):
      rimages = rimages.to(self.device)
      self.D.to(self.device)
      self.G.to(self.device)
      
      # Train the Discriminator
      fimages = self.trainD(rimages)
      # Train the Generator
      self.trainG(fimages)
      
      self.update_metrics()
      # Skipping the save_if_best() as there's no hard and fast rule which
      # can say looking at the losses whether the generator is performing better
      #self.save_if_best()


  def update_metrics(self):
    self.lossD(self.dloss)
    self.lossG(self.gloss)
    

  def store_details(self, part):
    self.train_list.append((self.lossD.get_curr_slot_avg(),
                            self.lossG.get_curr_slot_avg()))
  
  def show_epoch_details(self, e):
      print ('Epoch: {}/{} - DLoss: {:3f} - GLoss: {:3f}'.format(e, self.epochs,
             self.train_list[-1][0], self.train_list[-1][1]))

  def save_epoch_model(self, e):
      torch.save({'net' : self.G.state_dict(),
                  'optim' : self.optG.state_dict()},
                  'G-{}-{}'.format(e, self.train_list[-1][1]))


  def generate(self):
    z = self.create_random_noise(1)
    self.G.eval()
    img = self.G(z).detach()
    imshow(img.squeeze())

