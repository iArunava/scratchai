import numpy as np
from .BaseLoss import BaseLoss

class MSELoss(BaseLoss):

    def __init__(self):
        '''
        Constructor class for the Mean Squared Error Loss
        '''
        pass

    def calculate(self, outputs, targets):
        '''
        Method to calculate the Mean Squared Error Loss

        Arguments:
        - outputs - a predicted value, can be an integer or an array
        - targets - the true value, can be an integer or an array

        Returns:
        - integer - an integer for the mean squared error loss between the outputs and targets
        '''
        

        loss = ((outputs - targets) ** 2) / 2
        return loss
