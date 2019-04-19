"""
The Saliency Map Method Attack
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from scratchai.attacks.attacks.attack import Attack

class SaliencyMapMethod(Attack):
  """
  The Jacobian-based Saliency Map Method (Papernot et al. 2016)
  Paper Link: https://arxiv.org/pdf/1511.07528.pdf

  Arguments
  ---------
  model : nn.Module
      The model on which the attack needs to be performed.
  dtype : str
      The data type of the model.

  Returns
  -------
  adv : torch.tensor
     The adversarial Example of the input.
  """

  def __init__(self, model, dtype='float32', **kwargs):
    super().__init__(model, dtype, **kwargs)

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.

    Arguments
    ---------
    x : torch.tensor
      The input to the model
    kwargs : dict
        Additonal arguments
    """

    # Parse and save attack specific parameters
    assert self.parse_params(**kwargs)

    if self.y_target is None:
      # TODO torch.autograd.grad doesn't support batches
      # So, revise the implementation when it does in future releases
      def random_targets(gt):
        result = gt.clone()
        classes = gt.shape[1]
        # TODO Remove the blank () after #18315 in pytorch
        return torch.roll(result, int(torch.randint(nb_classes, ())))
      
      labels, nb_classes = self.get_or_guess_labels(x, kwargs)
      self.y_target = random_targets(labels)
      self.y_target = self.y_target.view([1, nb_classes])
      #print (torch.argmax(self.y_target, dim=1))
    
    # TODO Remove this line, just there for debug purpose
    x_adv = x
    # TODO Uncomment and check
    #x_adv = jsma_symbolic(x, y_target, model, theta, gamma, clip_min,
               #clip_max)
    return x_adv

  def parse_params(self, theta=1., gamma=1., clip_min=0., clip_max=1.,
          y_target=None):
    """
    Takes in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    Attack-specific parameters:
    ---------------------------
    theta : float, optional
        Perturbation introduced to modified components
        (can be positive or negative). Defaults to 1.
    gamma : float, optional
        Maximum percentage of perturbed features. Defaults to 1.
    clip_min : float, optional
         Minimum component value for clipping
    clip_max : float, optional
         Maximum component value for clipping
    y_target : torch.tensor, optional
         Target tensor if the attack is targetted
    """

    self.theta = theta
    self.gamma = gamma
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.y_target = y_target
    
    return True


def jsma_symbolic(x, y_target, net, theta, gamma, clip_min, clip_max):
  """
  PyTorch Implementation of the JSMA (see https://arxiv.org/abs/1511.07520
  for the details about the algorithm design choices).

  Arguments
  ---------
  x : torch.tensor
    The input to the model
  y_target : torch.tensor
       The target tensor
  model : nn.Module
      The pytorch model
  theta : float
      delta for each feature adjustment.
  gamma : float
      a float between 0 and 1 indicating the maximum distortion
      percentage.
  clip_min : float
       minimum value for components of the example returned
  clip_max : float
       maximum value for components of the example returned.
  
  Returns
  -------
  x_adv : torch.tensor
      The adversarial example.
  """

  classes = int(y_target.shape[1])
  features = int(np.product(x.shape[1:]))

  max_iters = np.floor(features * gamma / 2)
  increase = bool(theta > 0)

  tmp = np.ones((features, features), int)
  np.fill_diagonal(tmp, 0)

  # Compute the initial search domain. We optimize the initial search domain
  # by removing all features that are already at their maximum values
  # (if increasing input features -- otherwise, at their minimum value).
  if increase:
    search_domain = (x < clip_max).view(-1, features)
  else:
    search_domain = (x > clip_min).view(-1, features)
  
  # TODO remove this
  max_iters = 1

  while max_iters:
    logits = net(x)
    preds  = torch.argmax(logits, dim=1)
  
    # Create the Jacobian Graph
    list_deriv = []
    for idx in range(classes):
      deriv = grad(logits[:, idx], x)
      print (deriv)
      list_deriv.append(deriv)

    grads = (torch.stack(list_deriv, dim=0).view(classes, -1, features))

    max_iters -= 1
