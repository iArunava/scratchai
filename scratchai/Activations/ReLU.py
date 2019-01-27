import numpy as np

class ReLU(BaseActivation):
    
    def __init__(self):
        '''
        Constructor for the ReLU class
        '''
        self.name = 'relu'

    def forward(self, x):
        x = np.clip(x, a_min=0, a_max=None)
        return x
