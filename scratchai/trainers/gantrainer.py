"""
This file stores the code that allows to train GANs.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2

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
    #self.label = torch.full((self.batch_size,), self.real, device=self.device)
    
    # Setting up the optimizers
    assert isinstance(self.optimizer, tuple) and len(self.optimizer) == 2
    print ('[INFO] The first optimizer is assumed to be for the Generator!')
    self.optG, self.optD = self.optimizer
    
    self.lossD = AvgMeter('Discriminator Loss')
    self.lossG = AvgMeter('Generator Loss')

    self.train_list = []

    self.z_size = 100

    print (self.G, self.D)
    print (self.optG, self.optD)
  

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

  

  def before_train(self):
    self.D.to(self.device)
    self.G.to(self.device)


  def trainD(self, data):
    real = self.D(data)
    label = torch.full((self.batch_size,), self.real, device=self.device)
    rloss = self.criterion(real, label)

    #self.optD.zero_grad()
    
    # TODO Check whether this can be made to pass backward at once with rloss
    # at the end, that is keep both the backwards together with opt.step
    # if so, abstract the whole thing into the updateD() function
    #rloss.backward()
    
    # Create random noise to be passed to the generator
    z = self.create_random_noise()
    
    fimages = self.G(z)
    label = torch.full((self.batch_size,), self.fake, device=self.device)
    # NOTE This detach is very much needed for things to work!
    # Its the first Big Bug I faced while coding DCGAN
    fake = self.D(fimages.detach())
    floss = self.criterion(fake, label)
    #floss.backward()

    self.dloss = rloss + floss

    self.optD.zero_grad()
    self.dloss.backward()
    self.optD.step()

    return fimages
    
    
  def trainG(self, data):
    label = torch.full((self.batch_size,), self.real, device=self.device)
    fake = self.D(data)
    self.gloss = self.criterion(fake, label)
    # TODO Move the below two lines in updateG()
    self.optG.zero_grad()
    self.gloss.backward()
    self.optG.step()
   
  
  def train_body(self):
    for ii, (rimages, _) in enumerate(tqdm(self.train_loader)):
      rimages = rimages.to(self.device)

      # Rescale the images
      #rimages = rimages * 2 - 1
      
      # Train the Discriminator
      fimages = self.trainD(rimages)
      # Train the Generator
      self.trainG(fimages)
      
      self.update_metrics()
      # Skipping the save_if_best() as there's no hard and fast rule which
      # can say looking at the losses whether the generator is performing better
      #self.save_if_best()


  def update_metrics(self):
    self.lossD(self.dloss.data.cpu(), self.batch_size)
    self.lossG(self.gloss.data.cpu(), self.batch_size)
    

  def store_details(self, part):
    self.train_list.append((self.lossD.get_curr_slot_avg(),
                            self.lossG.get_curr_slot_avg()))
  
  def show_epoch_details(self, e):
      print ('Epoch: {}/{} - DLoss: {:3f} - GLoss: {:3f}'.format(e+1, self.epochs,
             self.train_list[-1][0], self.train_list[-1][1]))

  def save_epoch_model(self, e):
      torch.save({'netG' : self.G.state_dict(),
                  'netD' : self.D.state_dict(),
                  'optim' : self.optG.state_dict()},
                  'GD-{}-{:.3f}.pth'.format(e, self.train_list[-1][1]))


  def generate(self, save:bool=False):
    z = self.create_random_noise(1)
    self.G.eval()
    img = self.G(z).detach()
    if not save: imshow(img.data.cpu().squeeze())
    else: cv2.imwrite('r.png', img.squeeze().detach().cpu().numpy())

  def plot_gloss_vs_dloss(self):
    dlosses = list(map(lambda x : x[0], self.train_list))
    glosses = list(map(lambda x : x[1], self.train_list))
     
    epochs = np.arange(1, self.epochs_complete+1)
    plt.plot(epochs, dlosses, 'b', label='Discriminator Loss')
    plt.plot(epochs, dlosses, 'o', label='Generator Loss')
    plt.xlabel('Epochs')
    plt.ylable('Loss')
    plt.legend()
    plt.show()
