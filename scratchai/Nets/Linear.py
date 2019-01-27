import numpy as np
from BaseModel import BaseModel

class Linear(BaseModel):
    '''
    Class for the general purpose Linear Dense Feed Forward Network.
    '''

    def __init__(self, in_units, out_units, initialization='zero'):
        '''
        The constructor function for the Linear Class.

        Arguments:
        - in_units : the number of in_units for a Linear architecture
        - out_units : the number of out_units for a Linear architecture

        '''
        
        super().__init__()

        self.in_units = in_units
        self.out_units = out_units
        self.weights = np.zeros((in_units, self.out_units))

    def forward(self, x):
        '''
        The forward function of the Linear

        Arguments:
        - x : a vector of shape [self.in_units, 1]

        Returns:
        - x : a vector of shape [self.out_units, 1]
        '''
        x = np.dot(x, self.weights)
        return x
