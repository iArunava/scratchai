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
SAL = 'saliency_map_method'
FGM = 'fgm'
PGD = 'pgd'

class TestAttacks(unittest.TestCase):
  
  # TODO Shorten the url
  url = 'https://www.publicdomainpictures.net/pictures/210000/nahled/tiger-in-the-water-14812069667ks.jpg'
  trf = transforms.Compose([transforms.Resize(256),
               transforms.CenterCrop(224),
               transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225])
              ])
  trf2 = transforms.Compose([transforms.Resize(130),
               transforms.CenterCrop(128),
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
      # TODO No need to call Noise again and again in each iteration
      self.check_atk(net, img, scratchai.attacks.noise, t=NOISE)
      print ('[INFO] Attack worked successfully!')
      del net
  
  def test_smm_atk(self):
    """
    Tests to check that saliency map method is working as expected.
    """
    self.init()
    img = TestAttacks.img

    all_models = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 
                  'resnet101', 'resnet152']
    net = getattr(models, 'alexnet')(pretrained=True)
    self.check_atk(net, img, scratchai.attacks.smm, t=SAL)

    '''
    for model in all_models:
      print ('[INFO] Testing Noise attack on {}'.format(model))
      net = getattr(models, model)(pretrained=True)
      atk = scratchai.attacks.SaliencyMapMethod(net)
      self.check_atk(net, img, atk)
      print ('[INFO] Attack worked successfully!')
      del net, atk
    '''
    
  def test_semantic(self):
    """
    tests to ensure semantic attack works!
    """
    self.init()
    img = TestAttacks.img

    # TODO replace this with a scratchai model
    all_models = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    for model in all_models:
      print ('[INFO] Testing Semantic attack on {}'.format(model))
      net = getattr(models, model)(pretrained=True)
      # TODO No need to call Noise again and again in each iteration
      self.check_atk(net, img, scratchai.attacks.semantic, t=SEMANTIC)
      print ('[INFO] Attack worked successfully!')
      del net

  def init(self):
    """
    Contains operations needed to perform before
    any test in this class is executed.
    """
    if TestAttacks.img is None:
      with open('/tmp/test.png', 'wb') as f:
        f.write(requests.get(TestAttacks.url).content)
      TestAttacks.img = Image.open('/tmp/test.png')
  
  def test_pgd(self):
    """
    tests to ensure that fast_gradient_method attack works
    """
    with open('/tmp/test.png', 'wb') as f:
      f.write(requests.get(TestAttacks.url).content)
    img = Image.open('/tmp/test.png')

    # TODO replace this with a scratchai model
    # Running inference on just alexnet as it takes too long, otherwise
    # TODO Update other tests to just infer with one model for quick testing and 
    # Maybe an option to perform the rigourous testing, if needed.
    #all_models = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    all_models = ['alexnet']
    for model in all_models:
      to_pred = int(torch.randint(1000, ()))
      print ('[INFO] Testing PGD attack on {}'.format(model))
      net = getattr(models, model)(pretrained=True)
      self.check_atk(net, img, scratchai.attacks.pgd, t=PGD, y=torch.tensor([to_pred]))
      print ('[INFO] Attack worked successfully!')
      del net

  def test_fgm(self):
    """
    tests to ensure that fast_gradient_method attack works
    """
    with open('/tmp/test.png', 'wb') as f:
      f.write(requests.get(TestAttacks.url).content)
    img = Image.open('/tmp/test.png')

    # TODO replace this with a scratchai model
    # TODO write tests for targeted attack with fgm
    # TODO Update other tests to just infer with one model for quick testing and 
    # Maybe an option to perform the rigourous testing, if needed.
    #all_models = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    all_models = ['alexnet']
    for model in all_models:
      print ('[INFO] Testing FGM attack on {}'.format(model))
      net = getattr(models, model)(pretrained=True)
      self.check_atk(net, img, scratchai.attacks.fgm, t=FGM)
      print ('[INFO] Attack worked successfully!')
      del net


  def scale(self, img):
    return img * (255. / img.max())

  def check_atk(self, net, img, atk, t, y=None):
    # Get true pred
    net.eval()
    true_pred = int(torch.argmax(net(TestAttacks.trf(img).unsqueeze(0)), dim=1))
    
    # Adversarial Example
    if t == NOISE:
      adv_x = atk(torch.from_numpy(np.array(img))).transpose(2, 1).transpose(1, 0)
      adv_pred = int(torch.argmax(net(adv_x.unsqueeze(0)), dim=1))
    elif t == SEMANTIC:
      img = TestAttacks.trf(img)
      #img = self.scale(img) # It works w/ as well as w/o scaling.
      adv_x = atk(img)
      #plt.imshow(adv_x.transpose(0, 1).transpose(1, 2)); plt.show()
      adv_x = TestAttacks.trf(transforms.ToPILImage()(adv_x))
      adv_pred = int(torch.argmax(net(adv_x.unsqueeze(0)), dim=1))
    elif t == FGM or t == PGD:
      img = TestAttacks.trf(img)
      adv_x = atk(net, img.unsqueeze(0), y=y)
      adv_pred = int(torch.argmax(net(adv_x), dim=1))
    elif t == SAL:
      # TODO Passing noise, just for testing. Remove it.
      img = torch.randn(3, 64, 64)
      #img = TestAttacks.trf2(img)
      adv_x = atk(net, img.unsqueeze(0), y=y)
      adv_pred = int(torch.argmax(net(adv_x), dim=1))

    print (true_pred, adv_pred)
    self.assertFalse(true_pred == adv_pred, 'The attack doesn\'t work!')
    if y is not None:
      self.assertTrue(adv_pred == int(y), 'The attack doesn\'t work!')
      print ('TARGETED ATTACK WORKS!')
