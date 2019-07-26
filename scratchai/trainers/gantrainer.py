"""
This file stores the code that allows to train GANs.
"""

import torch
import torch.nn as nn

from scratchai.trainers.trainer import Trainer


__all__ = ['GANTraniner']


class GANTraniner(Trainer):
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
    
    # Setting up the optimizers
    assert isinstance(self.opt, tuple) and len(tuple) == 2
    print ('[INFO] The first optimizer is assumed to be for the Generator!')
    self.optG, self.optD = self.opt
    
    self.lossD = AvgMeter('Discriminator Loss')
    self.lossG = AvgMeter('Generator Loss')
  

  def before_epoch_start():
    self.lossD.create_and_shift_to_new_slot()
    self.lossG.create_and_shift_to_new_slot()


  def fit_body(self):
    self.before_epoch_start()
    self.train()

  
  def trainD(self, data):
    real = self.D(data)
    label = torch.full((self.batch_size,), self.real, device=self.device)
    loss = self.criterion(real, label)
    
    # NOTE This was at the start, make sure, this doesn't change things
    self.optD.zero_grad()


  def train_body(self):
    raise NotImplementedError
    for ii, (rimages, _) in enumerate(tqdm(self.tran_loader)):
      rimages = rimages.to(self.device)
      
      # Train the Discriminator
      self.trainD(rimages)


    
