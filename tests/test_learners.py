import unittest
import torch.nn as nn
import scratchai

class TestLearners(unittest.TestCase):
    
    def test_summary_exec(self):
        '''
        This test just ensures the learner.summary is working 
        and it doesn't raise Exceptions
        '''
        net = scratchai.UNet(3, 4)
        l = scratchai.learners.Learner(net)
        l.summary()
