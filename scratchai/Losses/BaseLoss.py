import autograd.numpy as np

class BaseLoss(object):
    '''
    Base Class for all the loss functions
    '''
    def calculate(self):
        pass

    def __call__(self, outputs, targets):
        return self.calculate(outputs, targets)

    def backward(self):
        pass
