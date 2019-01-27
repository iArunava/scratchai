import numpy as np

class BaseModel(object):
    '''
    The base model for all the Nets
    '''
    def forward(self):
        pass

    def __call__(self, x):
        return self.forward(x)
