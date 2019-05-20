import scratchai
import torch
import torch.nn as nn
import unittest
import requests
import zipfile
import io
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

from scratchai import *

NOISE = 'noise'
SEMANTIC = 'semantic'
FGM = 'fgm'
PGD = 'pgd'
DEEPFOOL = 'deepfool'

class TestAttacks(unittest.TestCase):
  
  # TODO Shorten the url
  url = 'https://www.publicdomainpictures.net/pictures/210000/nahled/tiger-in-the-water-14812069667ks.jpg'
  trf = imgutils.get_trf('rz256_cc224_tt_normimgnet')
  
  url_dset = 'https://www.dropbox.com/s/6bg8ntqcs4r98i9/testdataset.zip?dl=1'

  def test_noise_atk(self):
    """
    Tests to check that the Noise Attack works
    """
    with open('/tmp/test.png', 'wb') as f:
      f.write(requests.get(TestAttacks.url).content)
    img = Image.open('/tmp/test.png')

    #all_models = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    all_models = ['alexnet']
    for model in all_models:
      print ('[INFO] Testing Noise attack on {}'.format(model))
      net = getattr(nets, model)(pretrained=True)
      # TODO No need to call Noise again and again in each iteration
      self.check_atk(net, img, scratchai.attacks.noise, t=NOISE)
      print ('[INFO] Attack worked successfully!')
      del net

  def test_semantic(self):
    """
    tests to ensure semantic attack works!
    """
    with open('/tmp/test.png', 'wb') as f:
      f.write(requests.get(TestAttacks.url).content)
    img = Image.open('/tmp/test.png')

    # Maybe an option to perform the rigourous testing, if needed.
    #all_models = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    all_models = ['alexnet']
    for model in all_models:
      print ('[INFO] Testing Semantic attack on {}'.format(model))
      net = getattr(nets, model)(pretrained=True)
      # TODO No need to call Noise again and again in each iteration
      self.check_atk(net, img, scratchai.attacks.semantic, t=SEMANTIC)
      print ('[INFO] Attack worked successfully!')
      del net
  
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
  
  def test_deepfool(self):
    """
    tests to ensure that DeepFool attack works.
    """
    with open('/tmp/test.png', 'wb') as f:
      f.write(requests.get(TestAttacks.url).content)
    img = Image.open('/tmp/test.png')

    all_models = ['alexnet']
    for model in all_models:
      print ('[INFO] Testing DeepFool attack on {}'.format(model))
      net = getattr(nets, model)(pretrained=True)
      self.check_atk(net, img, scratchai.attacks.deepfool, t=DEEPFOOL)
      print ('[INFO] Attack worked successfully!')
      del net
    
  def test_benchmark(self):
    net = nets.resnet18()
    if not os.path.exists('/tmp/imgnet/'):
      r = requests.get(TestAttacks.url_dset)
      z = zipfile.ZipFile(io.BytesIO(r.content))
      z.extractall('/tmp/')

    atks = [attacks.Noise, attacks.Semantic, attacks.FGM, attacks.PGD, 
            attacks.DeepFool]
    for atk in atks: attacks.benchmark_atk(atk, net, root='/tmp/imgnet/')

  def scale(self, img):
    return img * (255. / img.max())

  def check_atk(self, net, img, atk, t, y=None):
    # Get true pred
    net.eval()
    true_pred = int(torch.argmax(net(TestAttacks.trf(img).unsqueeze(0)), dim=1))
    
    # Adversarial Example
    if t == NOISE:
      adv_x = atk(torch.from_numpy(np.array(img)).float()).transpose(2, 1).transpose(1, 0)
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
      adv_x = atk(img.unsqueeze(0), net, y=y)
      adv_pred = int(torch.argmax(net(adv_x), dim=1))
    elif t == DEEPFOOL:
      img = TestAttacks.trf(img)
      adv_x = atk(img.unsqueeze(0), net)
      adv_pred = int(torch.argmax(net(adv_x), dim=1))
      
    print (true_pred, adv_pred)
    self.assertFalse(true_pred == adv_pred, 'The attack doesn\'t work!')
    if y is not None:
      self.assertTrue(adv_pred == int(y), 'The attack doesn\'t work!')
      print ('TARGETED ATTACK WORKS!')
