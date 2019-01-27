import numpy as np

class BaseActivation(object):
    '''
    Base class for all the activations
    '''
    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)
