import scratchai
import torch
import torch.nn as nn
import unittest
import requests
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

NOISE = 'noise'
SEMANTIC = 'semantic'

class TestAttacks(unittest.TestCase):
  
  # TODO Shorten the url
  url = 'https://www.publicdomainpictures.net/pictures/210000/nahled/tiger-in-the-water-14812069667ks.jpg'
  trf = transforms.Compose([transforms.Resize(256),
               transforms.CenterCrop(224),
               transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225])
              ])
  img = None

  def test_noise_atk(self):
    """
    Tests to check that the Noise Attack works
    """
    self.init()
    img = TestAttacks.img

    # TODO replace this with a scratchai model
    all_models = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 
                  'resnet101', 'resnet152']

    for model in all_models:
      print ('[INFO] Testing Noise attack on {}'.format(model))
      net = getattr(models, model)(pretrained=True)
      atk = scratchai.attacks.Noise(net)
      self.check_atk(net, img, atk)
      print ('[INFO] Attack worked successfully!')
      del net, atk
  
  def test_saliency_map_method_atk(self):
    """
    Tests to check that saliency map method is working as expected.
    """
    self.init()
    img = TestAttacks.img

    all_models = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 
                  'resnet101', 'resnet152']

    net = getattr(models, 'alexnet')(pretrained=True)
    atk = scratchai.attacks.SaliencyMapMethod(net)
    self.check_atk(net, img, atk)

    '''
    for model in all_models:
      print ('[INFO] Testing Noise attack on {}'.format(model))
      net = getattr(models, model)(pretrained=True)
      atk = scratchai.attacks.SaliencyMapMethod(net)
      self.check_atk(net, img, atk)
      print ('[INFO] Attack worked successfully!')
      del net, atk
    '''
    
  def init(self):
    """
    Contains operations needed to perform before
    any test in this class is executed.
    """
    if TestAttacks.img is None:
      with open('/tmp/test.png', 'wb') as f:
        f.write(requests.get(TestAttacks.url).content)
      TestAttacks.img = Image.open('/tmp/test.png')

  def test_semantic(self):
    """
    tests to ensure semantic attack works!
    """
    with open('/tmp/test.png', 'wb') as f:
      f.write(requests.get(TestAttacks.url).content)
    img = Image.open('/tmp/test.png')

    # TODO replace this with a scratchai model
    all_models = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    for model in all_models:
      print ('[INFO] Testing Semantic attack on {}'.format(model))
      net = getattr(models, model)(pretrained=True)
      atk = scratchai.attacks.Semantic(net)
      # TODO No need to call Noise again and again in each iteration
      self.check_atk(net, img, atk, t=SEMANTIC)
      print ('[INFO] Attack worked successfully!')
      del net, atk
  
  def scale(self, img):
    return img * (255. / img.max())

  def check_atk(self, net, img, atk, t):
    # Get true pred
    net.eval()
    true_pred = int(torch.argmax(net(TestAttacks.trf(img).unsqueeze(0)), dim=1))
    
    # Adversarial Example
    if t == NOISE:
      adv_x = atk.generate(torch.from_numpy(np.array(img))).transpose(2, 1).transpose(1, 0)
      adv_pred = int(torch.argmax(net(adv_x.unsqueeze(0)), dim=1))
    elif t == SEMANTIC:
      img = TestAttacks.trf(img)
      #img = self.scale(img) # It works w/ as well as w/o scaling.
      adv_x = atk.generate(img)
      #plt.imshow(adv_x.transpose(0, 1).transpose(1, 2)); plt.show()
      adv_x = TestAttacks.trf(transforms.ToPILImage()(adv_x))
      adv_pred = int(torch.argmax(net(adv_x.unsqueeze(0)), dim=1))
      
    print (true_pred, adv_pred)
    self.assertFalse(true_pred == adv_pred, 'The attack doesn\'t work!')
