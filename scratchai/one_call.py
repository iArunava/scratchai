import torch
import numpy as np

import os
import requests
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms as T
from matplotlib.gridspec import GridSpec
from scratchai.nets.style_transfer.image_transformation_net import ITN_ST
from scratchai.datasets.labels import *
from scratchai.pretrained import urls
from scratchai import *


__all__ = ['classify', 'stransfer', 'attack']


def classify(path:str, nstr='resnet18', trf:str=None, gray:bool=False):
  """
  One call to classify an image.

  Arguments
  ---------
  img : str
        The path to the image file / url to the image file.
  nstr : str, nn.Module
         The name of the net to be used in string format.
         or the net itself
  trf : str
        The transforms to be used in the image. 
        The str is passed to utils.get_trf(). 
        Defaults to 'rz256_cc224_tt_normimgnet'
  gray : bool
       Set this to True if the model expects 2d images. Defaults to False.
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
  assert type(trf) == str or type(trf) == type(None)
  if type(nstr) is str and nstr.endswith('_mnist'): gray = True
  
  if not trf:
    if not gray: trf = imgutils.get_trf('rz256_cc224_tt_normimgnet')
    else: trf = imgutils.get_trf('rz32_cc28_tt_normmnist')
  else: trf = imgutils.get_trf(trf)
  
  # Getting the net
  if type(nstr) is str: net = getattr(nets, nstr)()
  else: net = nstr
  net.eval()

  # Getting the image from `path`
  if type(path) == str:
    # Special Case: if net == 'lenet_mnist' then image needs to have one channel
    if gray: img = trf(imgutils.gray(path)).unsqueeze(0)
    # Normal Cases
    else: img = trf(imgutils.load_img(path)).unsqueeze(0)
  else:
    img = trf(path).unsqueeze(0)

  # Edge Case: In case of 2D Images, the above line adds the channel dim
  # And then the batch dim needs to be added.
  if len(img.shape) == 3: img = img.unsqueeze(0)
  
  # Getting the image from net
  val, pred = torch.max(net(img), dim=1)
  val = val.item(); pred = pred.item()
  
  # In case the net is trained on mnist
  if gray: label = mnist_labels[pred]
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

  trf = T.Compose([T.ToTensor(),
                   T.Lambda(lambda x : x.mul(255))])

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


def attack(x, atk=attacks.FGM, nstr='resnet18', ret:bool=False, **kwargs):
  """
  One call to perform an attack on an image.

  Arguments
  ---------
  x : str, torch.Tensor
      The input image.
  atk : scratchai.attack.attack
        The attack which to perform. Defaults to FGM attack.
  nstr : str
         The net to use. (if needed) Defaults to None.
        For black box attacks, no need of passing the net.
  ret : bool
        If true, it returns the original image, adversarial image and others
        like the true label, adversarial label and so on.
  """
  # Checks
  assert not atk == attacks.Noise, 'Noise attack not supported due to a diff '\
                                   'preprocessing pipeline for noise attacks.'\
                                   ' Will support in future'
  
  if 'y' in kwargs:
    if isinstance(kwargs['y'], int): 
      kwargs['y'] = torch.Tensor([kwargs['y']]).long()
    assert kwargs['y'].item() >= 0  and kwargs['y'].item() < 1000, 'The class'\
            'label must be between [0, 1000) where the number represents the'\
            'class label from the available classes in imagenet.'

  x = imgutils.load_img(x) if isinstance(x, str) else x
  tlabl = classify(x, nstr) if nstr is not None else classify(x)
  net = getattr(nets, nstr)().eval(); utils.freeze(net)
  atk = atk(net=net, **kwargs)

  # Preprocess image for attack
  trf = imgutils.get_trf('rz256_cc224_tt_normimgnet')
  advx = atk(trf(x).unsqueeze(0))

    
  ## TODO Uncomment these 2 lines and comment the 2 lines below these
  ## after figuring out how to convert the adversarial image to PIL 
  ## and retain the precision so the PIL image when converted back to tensor
  ## is misclassified correctly as the orginial adversarial image.
  #advx_pil = T.ToPILImage()(advx.squeeze())
  #plabl = classify(advx_pil, nstr) if nstr is not None else classify(advx_pil)
  val, pred = torch.max(net(advx), dim=1)
  plabl = (imagenet_labels[pred.item()], val.item())

  limgs = [trf(x), None, advx]
  limgs[1] = imgutils.diff_imgs(limgs[0], limgs[2])
  limgs[0] = imgutils.unnorm(limgs[0]); limgs[2] = imgutils.unnorm(limgs[2])
  ltils = ['Original Image\nPred: {} \nwith {:.1f} confidence.'
              .format(tlabl[0].split(',')[0], tlabl[1]), 'Difference', 
           'Adversarial Image\nPred: {} \nwith {:.1f} confidence.'
              .format(plabl[0].split(',')[0], plabl[1])]
  
  if ret: return limgs, tlabl, plabl

  # TODO Just go over the how the images are viz ones
  # in case something is weird.
  plt.figure(figsize=(10, 10))
  gs = GridSpec(1, 3)
  gs.update(wspace=0.025, hspace=0.0)
  
  for ii in range(3):
    plt.subplot(gs[ii])
    plt.axis('off')
    plt.title(ltils[ii])
    plt.imshow(imgutils.t2i(limgs[ii]))
  plt.show()
