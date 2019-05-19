import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import operator
import PIL
import os
import requests

from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from torchvision import transforms as T

from scratchai import utils


__all__ = ['thresh_img', 'mask_reg', 'mark_pnt_on_img', 'load_img', 't2i', 
           'imsave', 'imshow', 'unnorm', 'get_trf', 'surface_plot', 'gray',
           'diff_imgs', 'mean', 'std']


def thresh_img(img:np.ndarray, rgb, tcol:list=[0, 0, 0]):
  """
  This function is used to threshold the image to a certain color.

  Args:
    img (np.ndarray): The image
    rgb (tuple or list): The color below which all colors will be blacked.
    tcol (list): The color to be filled in the thresholded part of
          the image. Defaults to [0, 0, 0].

  Returns:
    np.ndarray: The threshold image
  """

  assert type(img) == np.ndarray
  assert type(rgb) == tuple or type(rgb) == list
  assert type(tcol) == list

  img = np.copy(img)
  tidx = (img[:, :, 0] < rgb[0]) \
     | (img[:, :, 1] < rgb[1]) \
     | (img[:, :, 2] < rgb[2])
  img[tidx] = tcol
  return img


def mask_reg(img, pnts, reln, deg:int=1, tcol:tuple=(0, 0, 0), 
      locate:bool=False, invert:bool=False) -> np.ndarray:
  """
  Region Masking.

  Given a set of points which corrsponds to a polygon this
  function masks that polygon and returns the masked image.
  
  Arguments
  ---------
  img : np.ndarray, shape (H, W, 3)
     The image which needs to be masked.
  pnts : list
     A list containing the set of points (x, y)
     where each is a list / tuple.
  reln : list
     Where each element is a list/tuple (n1, n2)
     where n1 < len(points) and n2 < len(points) and it denotes
     that these two points are connected. 
     Do note: that n1 corresponds to the n1th element in points.
  deg : int
     Degree of the fitting polynomial.
  tcol : tuple of 3 elements
     A list containing the rgb values for the
     color which the mask needs to be filled with.
     Defaults to (0, 0, 0)
  locate : bool
      Denotes if to mark the points also in the image. 
      Defaults to False.
  invert : bool
      Denotes whether to invert the final indexes. Defaults to False.

  Returns
  -------
  img : np.ndarray
     The masked image.
  """

  assert type(img) == np.ndarray
  for param in [pnts, reln, tcol]:
    assert type(param) == list or type(param) == tuple
  
  img = np.copy(img)
  h, w = img.shape[1], img.shape[0]
  ops = {'<' : operator.lt, '>' : operator.gt}
  
  # Fit the lines
  psolns = {}
  for ii, (p1, p2, rel) in enumerate(reln):
    pnt1 = pnts[p1]; pnt2 = pnts[p2]
    soln = list(np.polyfit((pnt1[0], pnt2[0]), (pnt1[1], pnt2[1]), deg))
    soln.append(rel)
    psolns[ii] = soln

  # Find the region inside the lines
  xx, yy = np.meshgrid(np.arange(h), np.arange(w))
  for ii, val in psolns.items():
    # TODO Give option to choose btw bitwise_and and bitwise_or
    reg_thresh = ops[val[2]](yy, (xx*val[0] + val[1])) if not ii else \
          reg_thresh & ops[val[2]](yy, (xx*val[0] + val[1]))
      
  # Color pixels which are inside the region of interest
  img[reg_thresh == False if invert else reg_thresh] = tcol
  
  return img if not locate else mark_pnt_on_img(img, pnts)


def mark_pnt_on_img(img, pnts:list, col:tuple=(0, 255, 0)) -> np.ndarray:
  """
  Mark points on Image.

  Arguments
  ---------
  img : np.ndarray - Shape like (H x W x 3)
     The image on which the points need to be marked.
  pnts : list of tuples
     where each tuple is (x, y) a point to mark.
  col : tuple of 3 elements
     which are the rgb values for the color of the point.
  Returns
  -------
  img : np.ndarray
     With all the points marked on the image
  """

  img = np.copy(img)
  for pnt in pnts:
    cv2.circle(img, pnt, 10, col, -1)
  return img


def load_img(path:str, rtype=PIL.Image.Image):
  """
  Helper function to load a image from url as well as path.

  Arguments
  ---------
  path : str
        The file path or url from where to load the image.
  
  rtype : data type object
          The type of object in which to return the img.

  Returns
  -------
  img : rtype
        where rtype is the type of object in which to return the object.
  """
  # Other types not supported
  assert rtype in [PIL.Image.Image, np.ndarray]

  # Check if url
  # TODO Improve check
  is_url = path.startswith('http')
  fname = path.split('/')[-1]
  if is_url:
    if not os.path.isfile('/tmp/{}'.format(fname)):
      with open('/tmp/'+fname, 'wb') as f:
        f.write(requests.get(path).content)
    path = '/tmp/'+fname
  
  img = Image.open(path).convert('RGB')
  
  if rtype is np.ndarray:
    return np.array(img)
  return img


def t2i(img, rt=PIL.Image.Image, no255=False):
  """
  Converts torch.Tensor images to PIL images.

  Arguments
  ---------
  img : torch.Tensor
        The tensor image which to convert.
  rt : type
       The type of Image to be returned
  Returns
  -------
  img : PIL.Image.Image
        The converted PIL Image

  Notes
  -----
  Expects a [1 x 3 x H x W] torch.Tensor or [3 x H x W] torch.Tensor
  Converts it to a PIL.Image.Image of [H x W x 3]

  Converting to PIL.Image.Image losses precision if source image
  is of type float
  """
  out= img.squeeze().transpose(0, 1).transpose(1, 2).detach().clone().cpu()
  if not no255: out = out.mul(255).clamp(0, 255)
  out = out.numpy()
  if rt == PIL.Image.Image:
    # Note: .astype('uint8') losses precision
    return Image.fromarray(out.astype('uint8'))
  elif rt == np.ndarray:
    return out


def imsave(img, fname='random.png'):
  """
  Helper function to save an image to disk.

  Arguments
  ---------
  img : PIL.Image.Image, torch.Tensor
        The image to save.
  
  fname : str
          File Name. Defaults to random.png
  """

  if isinstance(img, torch.Tensor): img = t2i(img)
  img.save(fname)


def imshow(img, normd:bool=False, rz=224, **kwargs):
  """
  Display image.

  Arguments
  ---------
  img : torch.Tensor, PIL.Image.Image, list
        The image to display
  normd : bool
          If True, and if img is torch.Tensor then it unnormalizes the image
          Defaults to False.
  rz : int, tuple
       Row and Cols to resize to. If int, its taken as (rz, rz)
       Defaults to 224.
  """
  if isinstance(rz, int): rz = (rz, rz)
  if isinstance(img, list):
    nimgs = len(img)
    fig = plt.figure(figsize=(8, 8))
    # TODO Update to handle when nimgs is a prime and a few other cases
    gp = utils.gpfactor(nimgs)
    odiv = nimgs // gp
    col, row = gp, odiv

    for i in range(1, row*col+1):
      fig.add_subplot(row, col, i)

      cimg = img[i-1]
      if isinstance(cimg, torch.Tensor):
        cimg = t2i(unnorm(cimg) if normd else cimg, **kwargs)
      if rz is not None: cimg = cimg.resize(rz, Image.ANTIALIAS)

      plt.axis('off'); plt.imshow(cimg)

  elif isinstance(img, torch.Tensor):
    img = t2i(unnorm(img) if normd else img, **kwargs)
    plt.imshow(img)

  else: plt.imshow(img)
  plt.show()


def unnorm(t:torch.Tensor, mean=None, std=None):
  """
  Given an image this unnormalizes the image and returns it

  Arguments
  ---------
  t : torch.Tensor
      The image to unnormalize
  mean : list,(in case of >1 channels) else float
         The mean to multiply to the image
         Defaults to: [0.485, 0.456, 0.406],
  std : list,(in case of >1 channels) else float
         The std to add to the image
         Defaults to: [0.229, 0.224, 0.225]

  Returns
  -------
  t : torch.Tensor
      The unnormalized image.
  """
  if mean is None: mean = torch.Tensor([0.485, 0.456, 0.406])
  if std is None: std = torch.Tensor([0.229, 0.224, 0.225])
  # Not sure how to change the dim while multiplying so
  # changing the channel dimension from 0 to 2
  # Performing the operations and then changing it back
  t = (t.squeeze().transpose(0, 1).transpose(1, 2) * std) + mean
  return t.transpose(2, 1).transpose(1, 0)


def mean(t):
  """
  Calculates and returns the mean of a tensor

  Arguments
  ---------
  t : torch.Tensor
      The tensor whose mean is to be calculated
  
  Returns
  -------
  m : float
      The mean of the tensor
  """
  if isinstance(t, torch.Tensor): t = t.numpy()
  return t.sum() / t.size


def std(t):
  """
  Calculates and returns the standard deviation of a tensor

  Arguments
  ---------
  t : torch.Tensor
      The tensor whose std is to be calculated
  
  Returns
  -------
  m : float
      The std of the tensor
  """
  if isinstance(t, torch.Tensor): t = t.numpy()
  return np.sqrt(((t - mean(t)) ** 2).sum() / t.size)


def gray(img):
  """
  Converts RGB Image to Grayscale.
  
  Arguments
  ---------
  img : str, PIL.Image.Image
        The Image which is to be grayscaled.

  Returns
  -------
  img : PIL.Image.Image
        The grayscaled Image.

  References
  ----------
  https://stackoverflow.com/a/12201744
  """

  if isinstance(img, str):
   img = load_img(img)
  img = np.array(img)
  return Image.fromarray(np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]))


def diff_imgs(img1, img2, show=False, **kwargs):
  """
  Returns the difference of 2 images.

  Arguments
  ---------
  img1 : torch.Tensor, [C x H x W]
         The image from which to be subtracted.
  img2 : torch.Tensor, [C x H x W]
         The image which to be substracted.
  """
  dimg = img1.float().squeeze() - img2.float().squeeze()
  if show: imshow(dimg, **kwargs)
  return dimg


def surface_plot(matrix:np.ndarray):
  """
  Function to make a surface plot for a 2D matirx.

  Arguemnts
  ---------
  matrix : np.array
           The matrix for which the surface needs to be plot.
  """
  x, y = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(x, y, matrix, cmap=plt.cm.coolwarm)
  #fig.colorbar(surf)
  plt.show()


def get_trf(trfs:str):
  """
  A function to quickly get required transforms.

  Arguments
  ---------
  trfs : str
         An str that represents what T are needed. See Notes

  Returns
  -------
  trf : torch.transforms
        The transforms as a transforms object from torchvision.

  Notes
  -----
  >>> get_trf('rz256_cc224_tt_normimgnet')
  >>> T.Compose([T.Resize(256),
                          T.CenterCrop(224),
                          T.ToTensor(),
                          T.Normalize([0.485, 0.456, 0.406], 
                                      [0.229, 0.224, 0.225])])
  """
  # TODO Write tests
  # TODO Add more options
  trf_list = []
  for trf in trfs.split('_'):
    if trf.startswith('rz'):
      val = (int(trf[2:]), int(trf[2:]))
      trf_list.append(T.Resize(val))
    elif trf.startswith('cc'):
      val = (int(trf[2:]), int(trf[2:]))
      trf_list.append(T.CenterCrop(val))
    elif trf.startswith('rr'):
      trf_list.append(T.RandomRotation(int(trf[2:])))
    elif trf.startswith('rc'):
      trf_list.append(T.RandomCrop(int(trf[2:])))
    # TODO Add other padding modes
    elif trf.startswith('pad'):
      trf_list.append(T.Pad(int(trf[3:]), padding_mode='reflect'))
    elif trf.startswith('rhf'):
      val = float(trf[3:]) if trf[3:].strip() != '' else 0.5
      trf_list.append(T.RandomHorizontalFlip(val))
    elif trf.startswith('rvf'):
      val = float(trf[3:]) if trf[3:].strip() != '' else 0.5
      trf_list.append(T.RandomVerticalFlip(val))
    # T.ColorJitter
    # TODO Add a way to specify all three values
    # As of we take just 1 value and pass 3 equal ones.
    elif trf.startswith('cj'):
      val = [float(trf[2:])] * 3
      trf_list.append(T.ColorJitter(*val))
    elif trf == 'tt':
      trf_list.append(T.ToTensor())
    elif trf == 'normimgnet':
      trf_list.append(T.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]))
    elif trf == 'normmnist':
      trf_list.append(T.Normalize((0.1307,), (0.3081,)))
    elif trf == 'fm255':
      trf_list.append(T.Lambda(lambda x : x.mul(255)))
    else:
      raise NotImplementedError

  return T.Compose(trf_list)
