import torch
import os
import requests
from PIL import Image
from torchvision import transforms
from scratchai.nets.clf.resnet import resnet18
from scratchai.datasets.labels import *
from scratchai.imgutils import *


__all__ = ['classify']


def classify(path:str):
  """
  One call to classify an image.

  Arguments
  ---------
  img : str
        The path to the image file / url to the image file.
  
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

  trf = transforms.Compose([transforms.Resize(256), 
                            transforms.CenterCrop(224), 
                            transforms.ToTensor(), 
                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                 [0.229, 0.224, 0.225])
                           ])
  img = trf(load_img(path))
  net = resnet18(pretrained=True).eval()
  pred = int(torch.argmax(net(img.unsqueeze(0)), dim=1))
  return imagenet_labels[pred]
