import numpy as np
import os
from PIL import Image

class DatasetLoader(object):

    def __init__(self, input_path, label_path):
        '''
        This is the Base DatasetLoader class

        INPUTS:
        - input_path : The path to the directory holding the training data
        - label_path : The path to the directory holding the label data
        '''
        
        # Define the class variables to get the 
        # input and label path
        self.input_path = input_path
        self.label_path = label_path

        # Check if the path names ends in '/'
        if self.input_path[-1] != '/':
            self.input_path += '/'
        if self.label_path[-1] != '/':
            self.label_path += '/'

        self.inpn = np.array(sorted(os.listdir(input_path)))
        self.labn = np.array(sorted(os.listdir(label_path)))

        self.tinp = len(self.inpn)
        self.tlab = len(self.labn)
        
    def show_paths(self):
        '''
        The method to show the input path

        Arguments:
        None

        Returns:
        - str = The input path
        - str = The label path
        '''

        return self.input_path, self.label_path

    def glowhigh(self, batch_size=1, return_range=True):
        if batch_size > self.total_inputs:
            raise RuntimeError('Batch size is greater than the total number of files present!!')

        oidx = np.random.randint(0, self.total_inputs, size=1)
        sidx = oidx + batch_size

        if sidx >= self.total_inputs:
            sidx = oidx
            oidx = sidx - batch_size
        
        if return_range:
            return np.arange(start=oidx, 
                             stop=sidx)

        return oidx, sidx

    def get_batch(self, batch_size):
        pass

    def create_loader(self):
        pass

    def check(self):
        # Make sure the input and label paths exists
        if not os.path.exists(self.input_path) or \
           not os.path.exists(self.label_path):
            raise RuntimeError('The path passed in doesn\'t exists!!')
        # Make sure there are same number of inputs as the number of labels
        assert (self.total_inputs == self.total_labels)
