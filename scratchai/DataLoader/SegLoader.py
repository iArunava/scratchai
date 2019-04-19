import numpy as np
import os
from scratchai.DataLoader.ImageLoader import ImageLoader
from . import color_code as colors
from PIL import Image
import glob
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision.transforms as trf
import cv2
import os
import subprocess

__all__ = ['SegLoader', 'camvid']

class SegLoader(ImageLoader):

  def __init__(self, ip:str, lp:str, nc:int, bs:int, trfs=None, imdf=True, d=None, cmap=None):
    """
    Constructor for the Segmentation Dataset Loader.

    Arguemnts
    ---------
    ip : str
         The input path
    lp : str
         The label path
    bs : int
         The batch size
    trfs : torchvision.transforms
           The transforms that needs to be performed on the Images
    imdf : bool
           If the image files are immediately in the paths mentioned
    dataset_is : str
                name of the dataset, if it is known then the color map will
                be loaded by default, without the need of passing
                Supported Datasets:
                  - CamVid : pass 'camvid' as value to this argument

    color_map : dict 
                a dictionary where the key is the class name
                and the value is a tuple or list with 3 elements
                one for each channel. So each key is a RGB value.
    """

    super().__init__(ip, lp)
    
    self.d = d
    if str(self.d) == colors.CAMVID:
      self.cmap = colors.camvid_color_map

    self.trfs = trfs
    if self.trfs is None:
      self.trfs = trf.Compose([trf.ToTensor()])
    
    self.d = d
    self.nc = nc
    self.colors = list(self.cmap.values())
    self.classes = list(self.cmap.keys())

    self.ip = ip if ip[0] == '/' else ip + '/'
    self.lp = lp if lp[0] == '/' else lp + '/'
    
    '''
    imdp = '**/*' if not imdf else '*'
    self.ipf = glob.glob(ip + imdp, recursive=True)
    self.lpf = glob.glob(lp + imdp, recursive=True)
    self.tinp = len(self.ipf)
    '''

    self.bs = bs
    self.len = len(self.inpn) // self.bs
    
    # FIXME Check if the self.get_batch() method works perfectly 
    # compared to pytoch implementation
    self.dltr = self.get_batch()
    '''
    # TODO: Update to use own loaders to support imdf
    ipd = torchvision.datasets.ImageFolder(ip, transform=self.trfs)
    self.dltr = DataLoader(ipd, batch_size=bs, shuffle=True, num_workers=2)
    lpd = torchvision.datasets.ImageFolder(lp, transform=self.trfs)
    self.dltt = DataLoader(lpd, batch_size=bs, shuffle=True, num_workers=2)
    '''
    
    # Check for unusualities in the given directory
    #self.check()

  def show_batch(self, t='fin'):
    """
    Arguments:
    :: t - ['inp', 'lab', 'com']
       inp - The inputs
       lab - The Labels
       com = Inputs and Labels combined

    """
    # Implicitly checks for self.y is not None
    assert self.x is not None
    assert t in ['inp', 'lab', 'fin']

    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 1)
    gs.update(wspace=0.5, hspace=0.5)

    for i in range(self.bs if self.bs <= 10 else 10):
      ax = plt.subplot(gs[i])
      plt.axis('off')
      inp = self.t2n(self.x[i])
      lab = self.decode(self.t2n(self.y[i], c=False))
      fin = cv2.addWeighted(inp, 0.5, lab, 0.5, 0, dtype=0)
      ax.imshow(locals()[t])

    plt.show()
  

  def t2n(self, t, c=True):
    if c:
      return t.transpose(0, 1).transpose(1, 2).detach().cpu().numpy()
    else:
      return t.detach().cpu().numpy()
      

  def one_batch(self):
    self.x, _ = next(iter(self.dltr))
    self.y, _ = next(iter(self.dltt))
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


  def decode(self, image=None):
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

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for label in range(0, self.nc):
      r[image == label] = self.colors[label][0]
      g[image == label] = self.colors[label][1]
      b[image == label] = self.colors[label][2]
    
    rgb = np.stack([r, g, b], axis=2)
    return rgb

  def __len__(self):
    return self.len

  def get_dltr(self):
    return self.dltr

  def get_dltt(self):
    return self.dltt

def camvid(root:str='.', download:bool=False, **kwargs):
  root = root + '/' if root[-1] is not '/' else root
  
  if download and not os.path.isdir(root + 'camvid/'):
    dirname = os.path.dirname(__file__)
    subprocess.run(['sh', dirname + '/get_datasets/get_camvid.sh', root])

  kwargs['ip'] = root + 'camvid/images/_images'
  kwargs['lp'] = root + 'camvid/labels/_labels'
  kwargs['d']  = 'camvid'
  kwargs['nc'] = 32

  assert os.path.isdir(kwargs['ip']) == True
  assert os.path.isdir(kwargs['ip']) == True

  return SegLoader(**kwargs)
