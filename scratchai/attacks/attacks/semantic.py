"""
Semantic adversarial Examples
"""

__all__ = ['semantic', 'Semantic']

def semantic(x, center:bool=True, max_val:float=1.):
  """
  Semantic adversarial examples.
  
  https://arxiv.org/abs/1703.06857

  Note: data must either be centered (so that the negative image can be
  made by simple negation) or must be in the interval of [-1, 1]

  Arguments
  ---------
  net : nn.Module, optional
        The model on which to perform the attack.
  center : bool
           If true, assumes data has 0 mean so the negative image is just negation.
           If false, assumes data is in interval [0, max_val]
  max_val : float
            Maximum value allowed in the input data.
  """

  if center:
    return x*-1
  return max_val - x

################################################################
###### Class to initialize this attack
###### mainly for the use with torchvision.transforms

class Semantic():
  def __init__(self, net=None, **kwargs):
    self.kwargs = kwargs
  def __call__(self, x):
    return semantic(x, **self.kwargs)
