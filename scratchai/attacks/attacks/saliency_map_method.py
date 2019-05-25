"""
The Saliency Map Method Attack
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

def smm(net, x, theta=1., y=None, gamma=1., clip_min=0., clip_max=1.):
  """
  The Jacobian-based Saliency Map Method (Papernot et al. 2016)
  Paper Link: https://arxiv.org/pdf/1511.07528.pdf

  Arguments
  ---------
  model : nn.Module
          The model on which the attack needs to be performed.
  x : torch.Tensor
      The input ot the model.
  y : torch.tensor, optional
      Target tensor if the attack is targetted
  theta : float, optional
          Perturbation introduced to modified components
          (can be positive or negative). Defaults to 1.
  gamma : float, optional
          Maximum percentage of perturbed features. Defaults to 1.
  clip_min : float, optional
             Minimum component value for clipping
  clip_max : float, optional
             Maximum component value for clipping

  Returns
  -------
  adv_x : torch.tensor
          The adversarial Example of the input.
  """

  if y is None:
    # TODO torch.autograd.grad doesn't support batches
    # So, revise the implementation when it does in future releases
    def random_targets(gt):
      result = gt.clone()
      classes = gt.shape[1]
      # TODO Remove the blank () after #18315 in pytorch is resolved.
      return torch.roll(result, int(torch.randint(nb_classes, ())))
    
    labels, nb_classes = get_or_guess_labels(net, x, y)
    y = random_targets(labels)
    y = y.view([1, nb_classes])
    #print (torch.argmax(y, dim=1))
  
  x.requires_grad = True
  x_adv = jsma_symbolic(x, y, net, theta, gamma, clip_min, clip_max)
  return x_adv


def get_or_guess_labels(net, x, y=None):
  """
  Get the label to use in generating an adversarial example for x.
  The kwargs are fed directly from the kwargs of the attack.
  If 'y' is in kwargs, then assume its an untargetted attack and use
  that as the label.
  If 'y_target' is in kwargs and is not None, then assume it's a 
  targetted attack and use that as the label.
  Otherwise, use the model's prediction as the label and perform an 
  untargetted attack.

  Returns
  -------
  labels : torch.tensor
           Return a 1-hot vector with 1 in the position of the class predicted.
  nc : int
       # Number of classes
  """

  if y is not None:
    labels = kwargs['y']
  else:
    logits = net(x if len(x.shape) == 4 else x.unsqueeze(0))
    # TODO Remove cls, not needed, just there for debugging purpose
    pred_max, _ = torch.max(logits, dim=1)
    #print (cls)
    labels = (logits == pred_max).float()
    labels.requires_grad = False
  
  return labels, labels.size(1)

def jsma_symbolic(x, y, net, theta, gamma, clip_min, clip_max):
  """
  PyTorch Implementation of the JSMA (see https://arxiv.org/abs/1511.07520
  for the details about the algorithm design choices).

  Arguments
  ---------
  x : torch.tensor
    The input to the model
  y : torch.tensor
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

  classes = int(y.shape[1])
  features = int(np.product(x.shape[1:]))
  #print (features)

  max_iters = np.floor(features * gamma / 2)
  increase = bool(theta > 0)

  zdiag = np.ones((features, features), int)
  np.fill_diagonal(zdiag, 0)

  # Compute the initial search domain. We optimize the initial search domain
  # by removing all features that are already at their maximum values
  # (if increasing input features -- otherwise, at their minimum value).
  if increase:
    search_domain = (x < clip_max).view(-1, features)
  else:
    search_domain = (x > clip_min).view(-1, features)
  
  # TODO remove this
  max_iters = 1
  net.eval()
  while max_iters:
    logits = net(x)
    preds  = torch.argmax(logits, dim=1)
    loss = nn.CrossEntropyLoss()(logits, preds)
    
    '''
    loss.backward()
    grads = x.grad
    print (grads.shape)
    '''

    # Create the Jacobian Graph
    list_deriv = []
    for idx in range(classes):
      #print (x.requires_grad)
      deriv = grad(logits[:, idx], x, retain_graph=True)[0]
      #print (deriv[0].shape, x.shape)
      list_deriv.append(deriv)

    #print (list_deriv[0].shape)
    #grads = (torch.stack(list_deriv, dim=0).view(classes, -1, features))
    grads = torch.stack(list_deriv, dim=0).view(classes, -1, features)
    print (grads.shape)
    #'''
    
    # Compute the Jacobian components
    # To help with the computation later, reshape the target_class
    # and other_class to [nb_classes, -1, 1].
    # The last dimension is added to allow broadcasting later.
    tclass = y.view(classes, -1, 1)
    oclass = (tclass == 0).float()
    print (tclass.shape, oclass.shape)
    
    # TODO Check the dim
    a = grads * tclass
    gtarget = torch.sum(grads * tclass, dim=0)
    gother = torch.sum(grads * oclass, dim=0)
    print (gtarget.shape, gother.shape)
    print (gtarget[:1])
    print (gother[:1])
    
    # Remove the already-used input features from the search space
    # Subtract 2 times the maximum value from those so that they
    # won't be picked later
    increase_coef = (4 * int(increase) - 2) * (search_domain == 0).float()

    target_tmp = gtarget
    target_tmp -= increase_coef * torch.max(torch.abs(gtarget), dim=1)
    target_sum = target_tmp.view(-1, features, 1) + target_tmp.view(-1, 1, features)

    other_tmp = gother
    other_tmp += increase_coef * torch.max(torch.abs(gother), dim=1)
    other_sum = other_tmp.view(-1, features, 1) * other_tmp.view(-1, 1, features)
    print ('BOOM')
    return 0

    # Create a mask to only keep features that match conditions
    if increase:
      scores_mask = ((target_sum > 0) & (other_sum < 0))
    else:
      scores_mask = ((target_sum < 0) & (other_sum > 0))

    # Create a 2D numpy array of scores for each pair of candidate features
    scores = scores_mask * (-target_sum * other_sum) *  zdiag

    # Extract the best 2 pixels
    best = torch.argmax(scores.view(-1, features*features), dim=1)

    p1 = best % features
    p2 = best // features
    p1_ohot = one_hot(p1, depth=features)
    p2_ohot = one_hot(p2, depth=features)

    # Check if more modification is needed
    # TODO preds is 1 hot vector in tf implementation
    mod_not_done = torch.sum(y * preds, dim=1) == 0
    cond = mod_not_done & (torch.sum(search_domain, dim=1) >= 2)

    # Update the search domain
    cond_float = cond.view(-1, 1)
    to_mod = (p1_ohot + p2_ohot) * cond_float

    search_domain = search_domain - to_mod

    # Apply the modifications to the image
    to_mod = to_mod.view(-1, *x.shape)
    if increase:
      x = torch.min(clip_max, x + to_mod * theta)
    else:
      x = torch.min(clip_min, x + to_mod * theta)

    max_iters -= 1

  return x
