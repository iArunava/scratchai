import unittest
import scratchai

class TestLoaders(unittest.TestCase):

    def test_seg(self):
        l = scratchai.SegLoader('./vault/camvid/images', './vault/camvid/labels', 2)
        self.assertEqual(type(l), scratchai.DataLoader.SegLoader.SegLoader, 'Type Mismatch')
