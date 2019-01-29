import numpy as np

class BaseLoss(object):
    '''
    Base Class for all the loss functions
    '''
    def calculate(self):
        pass

    def __call__(self):
        return self.calculate()
