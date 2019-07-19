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
    
