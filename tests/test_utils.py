import unittest
import numpy as np
from scratchai.imgutils import *

class TestImgUtils(unittest.TestCase):
    
    def test_thresh_img(self):
        """
        This test makes sure that the thresh_img function 
        is working as expected.
        """
        noise = np.random.randint(0, 255, size=(32, 32, 3))
        thresh = [45, 45, 45]
        img = thresh_img(noise, thresh, tcol=[255, 255, 255])

        self.assertEqual(type(img), np.ndarray, 'The return type is not np.ndarray')
        self.assertEqual(img.shape, (32, 32, 3), 'The shape of the output is' \
                                                    'not as expected')
        # TO make sure all pixels having value less than threshold is removed.
        # NOTE if tcol have values lower than thresh then values lower
        # than thresh will be present. That is why tcol = [255, 255, 255]
        # for the purposes of this test.
        self.assertFalse((img < thresh).any())
