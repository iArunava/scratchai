import unittest
import torch
import PIL
import numpy as np
import scratchai

from PIL import Image
from scratchai import imgutils
from scratchai import *



################################
### Check functions for imgutils
################################

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

  def test_avgmeter(self):
    name = 'name'; fmt = '.:2f'
    mtr = utils.AvgMeter(name, fmt)
    self.assertEqual(mtr.name, name, 'initialization went wrong!')
    self.assertEqual(mtr.fmt, fmt, 'initialization went wrong!')
    self.assertEqual(mtr.val + mtr.sum + mtr.cnt + mtr.avg, 0, 'result bad!')
    
    val = 0; cnt = 0
    for _ in range(torch.randint(0, 10, ())):
      val += torch.randint(0, 10, ()); cnt += torch.randint(0, 10, ())
      mtr(val/cnt, cnt)
      self.assertEqual(mtr.avg, (val / cnt), 'result bad!')

    self.assertEqual(str(mtr), '{} - {}'.format(name, val/cnt))

    

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
