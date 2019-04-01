import torch
import torch.nn as nn

class Learner(object):
    
    def __init__(self, model:nn.Module, loader, metrics:list=None, e:int=10):
        '''
        A Learner Object

        Arguments:
        :: model - The model to train
        :: loader - The DataLoader from where to get the data
        :: e - The number of epochs
        '''
        self.model = model
        self.loader = loader
        self.metrics = metrics
        self.e = e

    def fit(self):
        pass
