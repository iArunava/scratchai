import unittest
import torch
import torch.nn as nn
import PIL
import numpy as np
import scratchai

from PIL import Image
from scratchai import imgutils
from scratchai import *



#############################################
### Check functions for scratchai/imgutils.py
#############################################

class TestImgUtils(unittest.TestCase):
  
  url_1 = 'https://cdn.theatlantic.com/assets/media/img/mt/2015/10/' \
          '5594637726_3a0fdc9b7b_o/lead_720_405.jpg'

  def test_thresh_img(self):
    """
    This test makes sure that the thresh_img function 
    is working as expected.
    """
    noise = np.random.randint(0, 255, size=(32, 32, 3))
    thresh = [45, 45, 45]
    img = imgutils.thresh_img(noise, thresh, tcol=[255, 255, 255])

    self.assertEqual(type(img), np.ndarray, 'Return type not np.ndarray')
    self.assertEqual(img.shape, (32, 32, 3), 'The shape of output is' \
                          'not as expected')

    # TO make sure all pixels having value less than threshold is removed.
    # NOTE if tcol have values lower than thresh then values lower
    # than thresh will be present. That is why tcol = [255, 255, 255]
    # for the purposes of this test.
    self.assertFalse((img < thresh).any())

  def test_mask_reg(self):
    #TODO add tests for testing it further
    if not callable(getattr(imgutils, 'mask_reg', None)):
      raise NotImplementedError

  def test_mark_pnt_on_img(self):
    #TODO add tests for testing it further
    if not callable(getattr(imgutils, 'mark_pnt_on_img', None)):
      raise NotImplementedError
    
  def test_load_img(self):
    img = imgutils.load_img(TestImgUtils.url_1, np.ndarray)
    self.assertLessEqual(img.shape[-1], 3, '# of channels is not as expected')
    
  def test_t2i(self):
    noise = torch.randn(1, 3, 12, 12)
    image = imgutils.t2i(noise)
    self.assertEqual(type(image), PIL.Image.Image, 'Unexpected Type')
    image = np.array(image)
    self.assertEqual(list(image.shape), [12, 12, 3],'Unexpected shape')

  def test_imsave_imshow(self):
    #TODO add tests for testing it further
    utils.implemented(imgutils, 'imsave')
    utils.implemented(imgutils, 'imshow')

  def test_diff_imgs(self):
    t1 = torch.randn(3, 150, 150)
    t2 = torch.randn(3, 150, 150)
    self.assertTrue(imgutils.diff_imgs(t1, t1, normd=False).sum().item() == 0, 
               'nope!')
    self.assertTrue(imgutils.diff_imgs(t2, t2, normd=False).sum().item() == 0, 
               'nope!')
    self.assertFalse(imgutils.diff_imgs(t1, t2, normd=False).sum().item() == 0, 
               'nope!')
    
    # Explicit test
    t1 = torch.Tensor([[[1, 1, 1], [2, 2, 2], [3, 3, 3]]])
    t2 = torch.Tensor([[[2, 2, 2], [1, 1, 1], [4, 4, 4]]])
    t3 = torch.Tensor([[[-1, -1, -1], [1, 1, 1], [-1, -1, -1]]])
    t4 = torch.Tensor([[[1, 1, 1], [-1, -1, -1], [1, 1, 1]]])

    # With norm false
    dimg1 = imgutils.diff_imgs(t1, t2, normd=False)
    dimg2 = imgutils.diff_imgs(t2, t1, normd=False)
    self.assertTrue((dimg1 - t3.squeeze()).sum().item() == 0., 'nope!')
    self.assertTrue((dimg2 - t4.squeeze()).sum().item() == 0., 'nope!')

  def test_mean(self):
    for _ in range(torch.randint(1, 10, ())):
      t = torch.randint(0, 255, size=(3, 28, 28))
      tm = t.sum() / t.numel()
      m = imgutils.mean(t)
      self.assertEqual(tm, m, 'mean not correct!')
    
  def test_std(self):
    for _ in range(torch.randint(1, 10, ())):
      t = torch.randn(3, 28, 28)
      std = imgutils.std(t)
      self.assertEqual(round(std - torch.std(t).item(), 2), 0.0, 'nope!')


#############################################
### Check the functions in scratchai/utils.py
#############################################

class TestUtils(unittest.TestCase):
  
  def test_load_from_pth(self):
    #TODO add tests for testing it further
    if not callable(getattr(scratchai.utils, 'load_from_pth', None)):
      raise NotImplementedError

  def test_name_from_object(self):
    obj = nets.Lenet()
    name = utils.name_from_object(obj)
    self.assertTrue(name == 'lenet', 'doesn\t look good')
  
  def test_freeze(self):
    net = nets.resnet18()
    for p in net.parameters():
      self.assertTrue(p.requires_grad, 'initialization already frozen!')
    utils.freeze(net)
    for p in net.parameters():
      self.assertFalse(p.requires_grad, 'not working!')

  def test_topk(self):
    name = 'Acc'; cnt = 50
    for _ in range(torch.randint(1, 10, ())):
      topk = tuple(torch.randint(2, 10, size=(5,)).numpy())
      obj = utils.Topk(name, topk)
      topk = sorted(tuple(set(topk)))
      val = np.zeros(len(topk))

      for i in range(1, torch.randint(1, 10, size=())):
        curr_val = np.random.randint(0, 50, size=(len(topk),)).astype('float')
        obj.update(curr_val/cnt, cnt)
        val += curr_val
        k = np.random.choice(topk)
        self.assertEqual(round(obj.avgmtrs[name+str(k)].avg - 
                         val[topk.index(k)]/(cnt*i), 2), 0.0, 'nope!')
        
      self.assertEqual(len(obj.avgmtrs), len(topk), 'not working!')
      self.assertEqual(obj.ks, len(topk), 'not working!')
      # If len(topk) < 2: then a 0-d tensor will be passed which will throw exp
      if len(topk) > 2: self.assertRaises(AssertionError, 
                        lambda: obj.update(val[:len(topk)-1], cnt))
      
    topk = tuple(torch.randint(1, 10, size=(5,)).numpy())
    obj = utils.Topk(name, topk)
    self.assertEqual(name, obj.name, 'dude!')
    self.assertRaises(AssertionError, lambda: utils.Topk('A', (1, 3, 0)))

  def test_avgmeter(self):
    name = 'name'; fmt = '.:2f'
    mtr = utils.AvgMeter(name, fmt)
    self.assertEqual(mtr.name, name, 'initialization went wrong!')
    self.assertEqual(mtr.fmt, fmt, 'initialization went wrong!')
    self.assertEqual(mtr.val + mtr.sum + mtr.cnt + mtr.avg, 0, 'result bad!')
    
    val = 0.; cnt = 0.
    for _ in range(torch.randint(1, 10, ())):
      curr_val = torch.randint(1, 10, ()).float()
      curr_cnt = torch.randint(1, 10, ()).float()
      mtr(curr_val/curr_cnt, curr_cnt)
      val += curr_val; cnt += curr_cnt
      self.assertEqual(mtr.avg, (val / cnt), 'result bad!')

    self.assertEqual(str(mtr), '{} - {}'.format(name, val/cnt))

  def test_setatrib(self):
    # TODO Needs more tests
    net = nets.alexnet()
    val = nn.Identity()
    t = net.net[0]; utils.setatrib(net, 'net[0]', val)
    self.assertEqual(net.net[0], val, 'not working!')
    self.assertFalse(t == val, 'not working!')
  
  def test_count_params(self):
    net = nn.Conv2d(3, 4, 5, 2, 1)
    num = utils.count_params(net)
    self.assertEqual(num, (4*5*5*3) + 4, 'nope!')

  def test_sgdivisor(self):
    ulim = 100
    for n in np.random.randint(1, ulim, size=(5,)):
      s, g = utils.sgdivisor(int(n))
      self.assertTrue(n % s == 0, 'Nope!')
      self.assertTrue(n % g == 0, 'Nope!')
      # Explicit checks
      for ii in range(s-1, 1, -1):
        if ii == n: continue
        self.assertFalse(n % ii == 0, 'nope!')
      for ii in range(g+1, ulim):
        if ii == n: continue
        self.assertFalse(n % ii == 0, 'nope!')

  def test_gpfactor(self):
    pfs = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,
           79,83,89,97]
    # DO NOT CHANGE ulim AS pfs is a list of prime numbers b/w # 0 and 100.
    ulim = 100
    for n in np.random.randint(1, ulim, size=(5,)):
      gp = utils.gpfactor(int(n))
      self.assertTrue(n % gp == 0, 'Nope!')
      # Explicit checks
      for pf in pfs:
        if n % pf == 0: self.assertFalse(pf > gp , 'nope!')
    self.assertRaises(AssertionError, lambda: utils.gpfactor(0))

#############################################
### Check the functions in scratchai/attacks/utils.py
#############################################

class TestAtkUtils(unittest.TestCase):
  
  def test_optimize_linear(self):
    #TODO add tests for testing it further
    if not callable(getattr(scratchai.attacks.utils, 'optimize_linear', None)):
      raise NotImplementedError

  def test_clip_eta(self):
    #TODO add tests for testing it further
    if not callable(getattr(scratchai.attacks.utils, 'clip_eta', None)):
      raise NotImplementedError
