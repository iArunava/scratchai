from torchvision import transforms as T
from scratchai.attacks.attacks import *


'''
class FGM_TRF():
  def __init__(self,  
'''
def test_fgm(net:nn.Module, loader, **kwargs):
  """
  Benchmarks the Fast Gradient attack on the given loader.

  Arguments
  ---------
  net : nn.Module
        The net on which to test the attack
  loader : torch.utils.data.DataLoader
           The loader.
  
  Returns
  -------
  acc : float
        The accuracy of the net on the dataset.
  """

  net.eval()
  trf = T.Compose([T.Resize(256), T.CenterCrop(224), fgm(), ])
