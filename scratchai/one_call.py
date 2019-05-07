import torch
import numpy as np

import os
import requests

from PIL import Image
from torchvision import transforms
from scratchai.nets.style_transfer.image_transformation_net import ITN_ST
from scratchai.datasets.labels import *
from scratchai.pretrained import urls
from scratchai import *


__all__ = ['classify', 'stransfer']


def classify(path:str, nstr:str='resnet18', trf:str=None):
  """
  One call to classify an image.

  Arguments
  ---------
  img : str
        The path to the image file / url to the image file.
  nstr : str
         The name of the net to be used in string format.
  trf : str
        The transforms to be used in the image. 
        The str is passed to utils.get_trf(). 
        Defaults to 'rz256_cc224_tt_normimgnet'
  Returns
  -------
  pred : str
         The predicted value.

  Notes
  -----
  Given path to an image this function loads it and converts it to a tensor and
  normalizes it using imagenet specific normalization values and passes it to 
  a resnet output the predicted value.
  """

  trf = imgutils.get_trf('rz256_cc224_tt_normimgnet' if not trf else trf)
  
  # Getting the image from `path`
  if type(path) == str:
    # Special Case: if net == 'lenet_mnist' then image needs to have one channel
    if nstr.endswith('_mnist'): img = trf(imgutils.gray(path)).unsqueeze(0)
    # Normal Cases
    else: img = trf(imgutils.load_img(path)).unsqueeze(0)
  else:
    img = trf(path).unsqueeze(0)

  # Edge Case: In case of 2D Images, the above line adds the channel dim
  # And then the batch dim needs to be added.
  if len(img.shape) == 3: img = img.unsqueeze(0)
  
  # Getting the image from net
  net = getattr(nets, nstr)().eval()
  val, pred = torch.max(net(img), dim=1)
  val = val.item(); pred = pred.item()
  
  # In case the net is trained on mnist
  if nstr.endswith('_mnist'): label = mnist_labels[pred]
  # In case the net is trained on Imagenet
  else: label = imagenet_labels[pred]
  
  return label, val


def stransfer(path:str, style:str=None, save:bool=False, show:bool=True):
  """
  One call to transfer the style of an image.

  Arguments
  ---------
  img : str
        The path to the image file / url to the image file.
  style : str
          The style in which to convert. Defaults to None.
          If None, it picks up a style in random.
  save : bool
         If True, saves the image to disk. Default False.
  show : bool
         If true, it shows the image.
  Returns
  -------
  pred : str
         The predicted value.

  Notes
  -----
  Given path to an image this function loads it and converts it to a tensor and
  normalizes it using imagenet specific normalization values and passes it to 
  a resnet output the predicted value.
  """
  avbl_styles = ['elephant_skin', 'snake_skin']
  if style is None: style = np.random.choice(avbl_styles, 1)[0]
  sdict = utils.load_from_pth(getattr(urls, style + '_url'), style)

  trf = transforms.Compose([transforms.ToTensor(),
                            transforms.Lambda(lambda x : x.mul(255))])

  if type(path) == str:
    img = trf(imgutils.load_img(path))
  else:
    img = trf(path)

  net = ITN_ST(); net.load_state_dict(sdict); net.eval()
  out = net(img.unsqueeze(0)).squeeze().cpu()
  out = Image.fromarray(out.squeeze().transpose(0, 1).transpose(1, 2).detach()
             .clone().cpu().clamp(0, 255).numpy().astype('uint8'))
  if save: imgutils.imsave(out)
  if show: imgutils.imshow(out)
  return out
