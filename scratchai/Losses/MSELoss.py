import autograd.numpy as np
from autograd import elementwise_grad as egrad
from .BaseLoss import BaseLoss

class MSELoss(BaseLoss):

    def __init__(self):
        '''
        Constructor class for the Mean Squared Error Loss
        '''
        super().__init__()

        self.loss = 0
        self.grads = 0

    def calculate(self, outputs, targets):
        '''
        Method to calculate the Mean Squared Error Loss

        Arguments:
        - outputs - a predicted value, can be an integer or an array
        - targets - the true value, can be an integer or an array

        Returns:
        - integer - an integer for the mean squared error loss between the outputs and targets
        '''
        
        outputs = np.array(outputs)
        targets = np.array(targets)

        self.outputs = outputs
        self.targets = targets

        loss = ((outputs - targets) ** 2) / 2
        self.loss = loss
        return loss

    def backward(self):
        '''
        Method to calculate the gradients
        '''
        self.grads = egrad(self.calculate)(self.outputs, self.targets)
        return self.grads
