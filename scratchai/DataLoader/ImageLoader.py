import numpy as np
from PIL import Image
from scratchai.DataLoader.DatasetLoader import DatasetLoader
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as trfs

class ImageLoader(DatasetLoader):
    '''
    This is the class that helps you load inputs and labels for semantic
    segmentation datasets.

    '''

    def __init__(self, input_path, label_path):
        '''
        The constructor to the SegmentationDatasetLoader class

        Arguments:
        - input_path : The path to the directory for input images
        - label_path : The path to the directory for label images
        '''

        super().__init__(input_path, label_path)


    def get_batch(self, shuffle=True):
        '''
        Return a batch of the images from both inputs and labels

        Arguments:
        - sample_size = The number of images to return

        returns;
        - 2 - np.ndarray : (N x C x H x W)
                       where,
                       N - is the sample_size
                       C - the number of channels
                       H - the height of the image
                       W - the width of the image

        '''
        
        if shuffle:
            sample_idx = np.random.randint(low=0, 
                                   high=self.tinp, 
                                   size=self.bs)
        else:
            sample_idx = self.glowhigh(batch_size=self.bs,
                                  return_range=True)
            
        inputs = self.inpn[sample_idx]
        labels = self.labn[sample_idx]
        
        all_inps = []
        all_labs = []
        for inp, label in zip(inputs, labels):
            si = np.array(Image.open(self.input_path + inp))
            si = torch.tensor(si).transpose(2, 1).transpose(1, 0)

            li = np.array(Image.open(self.label_path + label))
            li = torch.tensor(li)

            all_inps.append(si)
            all_labs.append(li)

        self.x = torch.stack(all_inps, dim=0)
        self.y = torch.stack(all_labs, dim=0).squeeze()
        
        yield self.x, self.y
            
    def create_loader(self, batch_size=1, shuffle=True):
        while(1):
            yield self.get_batch(batch_size=batch_size,
                                 shuffle=True)

    def show_few(self, size=1, in_row=False, figsize=(20, 10), show_axis=False):

        # Check argument data types
        if not isinstance(show_axis, bool):
            raise RuntimeError('Unexpected data type passed for "show_axis" argument!')
        if not isinstance(in_row, bool):
            raise RuntimeError('Unexpected data type passed for "in_row" argument!')

        inputs, labels = self.get_batch(batch_size=size)
        
        rc_tuple = (size, 1)
        if in_row:
            rc_tuple = (1, size)

        if not show_axis:
            show_axis = 'off'
        else:
            show_axis = 'on'

        figure = plt.subplots(figsize=figsize)
        for ii in range(1, size+1):
            plt.subplot(*rc_tuple, ii)
            plt.imshow(inputs[ii-1])

        plt.show()
