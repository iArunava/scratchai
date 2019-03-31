import numpy as np
import os
from scratchai.DataLoader.ImageLoader import ImageLoader
from . import color_code as ccode
from PIL import Image
import glob
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision.transforms as trf

class SegLoader(ImageLoader):

    def __init__(self, ip:str, lp:str, bs:int, trfs=None, imdf=True, d=None, cmap=None):
        '''
        Constructor for the Segmentation Dataset Loader

        Arguemnts:
        :: ip - The input path
        :: lp - The label path
        :: bs - The batch size
        :: trfs - The transforms that needs to be performed on the Images
        :: imdf - If the image files are immediately in the paths mentioned
        :: dataset_is : name of the dataset, if it is known then the color map will
                       be loaded by default, without the need of passing
                       Supported Datasets:
                       - CamVid : pass 'camvid' as value to this argument

        :: color_map - dict - a dictionary where the key is the class name
                             and the value is a tuple or list with 3 elements
                             one for each channel. So each key is a RGB value.
        '''

        super().__init__(ip, lp)
        
        self.d = d
        if str(self.d) == ccode.CAMVID:
            self.cmap = ccode.camvid_color_map

        self.trfs = trfs
        if self.trfs is None:
            self.trfs = trf.Compose([trf.ToTensor()])
        
        self.d= d
        self.cmap = cmap
        '''
        self.classes = list(self.color_map.keys())
        self.colors = list(self.color_map.values())
        self.num_classes = len(color_map)
        '''

        self.ip = ip if ip[0] == '/' else ip + '/'
        self.lp = lp if lp[0] == '/' else lp + '/'
        
        '''
        imdp = '**/*' if not self.imdf else '*'
        self.ipf = glob.glob(ip + imdp, recursive=True)
        self.lpf = glob.glob(lp + imdp, recursive=True)
        '''
        self.bs = bs
        
        # TODO: Update to use own loaders to support imdf
        ipd = torchvision.datasets.ImageFolder(ip, transform=self.trfs)
        self.xloader = DataLoader(ipd, batch_size=bs, shuffle=True, num_workers=2)
        lpd = torchvision.datasets.ImageFolder(lp, transform=self.trfs)
        self.yloader = DataLoader(lpd, batch_size=bs, shuffle=True, num_workers=2)
        
        # Check for unusualities in the given directory
        #self.check()

    def show_batch(self):
        # Implicitly checks for self.y is not None
        assert self.x is not None

        plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=0.5, hspace=0.5)

        for i in range(self.bs if self.bs <= 10 else 10):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.imshow(self.t2n(self.x[i]))
            # use cv2.add_weighted

        plt.show()
    
    def t2n(self, t):
        return t.transpose(0, 1).transpose(1, 2).detach().cpu().numpy()

    def one_batch(self):
        self.x, _ = next(iter(self.xloader))
        self.y, _ = next(iter(self.yloader))
        return self.x, self.y
        
    def check(self):
        if self.dataset_is is None and self.color_map is None:
            raise RuntimeError('Both \'dataset_is\' and \'color_map\' can\'t be None')
        super(SegmentationDatasetLoader, self).check()
        

    def create_masks(self, image=None, path=None):
        '''
        A class that creates masks for each of the classes

        Arguments:
        - Image.PIL - Semantic Segmented Image where each pixel is colored
                      with a specific color
                      The Image is of size H x W x C
                      where H is the height of the image
                            W is the width of the image
                            C is the number of channels (3)

        Returns:
        - np.ndarray - of size N x H x W
                       where N is the number of classes
                             H is the height of the image
                             W is the width of the image
        '''
        
        if image is None and path is None:
            raise RuntimeError('Either image or path needs to be passed!')

        if not path is None:
            if not os.path.exists(path):
                raise RuntimeError('You need to pass a valid path!\n \
                                    Try passing a number if you are having trouble reaching \
                                    the filename')
            image = np.array(Image.open(path)).astype(np.uint8)

        if image.shape[-1] > 3 or image.shape[-1] < 3:
            raise RuntimeError('The image passed has more than expected channels!')
        
        masks = []
        for ii in self.color_map:
            color_img = []
            for j in range(3):
                color_img.append(np.ones((img.shape[:-1])) * ii[j])
            img2 = np.array(color_img, dtype=np.uint8).transpose(1, 2, 0)
            masks.append(np.uint8((image == img2).sum(axis=-1) == 3))

        return np.array(masks)


    def decode_segmap(self, image=None, path=None, image_num=None):
        '''
        The method helps one get a colorful image where each color corresponds to each class

        Arguments:
        - Image - np.array - A 2D Image where each pixel position is a number indicating
                             the class to which is belongs
        
        Returns:
        - np.array - H x W x C
                    where each pixel position [x, y, :]
                    is a color representing its RGB color which is passed in
                    with the color_map while initializing this class
        '''

        if image is None and path is None:
            raise RuntimeError('Either image or path needs to be passed!')

        if not path is None:
            if not os.path.exists(path):
                raise RuntimeError('You need to pass a valid path!\n \
                                    Try passing a number if you are having trouble reaching \
                                    the filename')
            image = Image.open(path)

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for label in range(0, self.num_classes):
            r[image == label] = self.colors[label][0]
            g[image == label] = self.colors[label][1]
            b[image == label] = self.colors[label][2]

        rgb = np.stack([r, g, b], axis=2)
        return rgb

def camvidloader(**kwargs):
    kwargs['d'] = 'camvid'
    return SegLoader(**kwargs)
