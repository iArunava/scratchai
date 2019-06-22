"""
The metrics used to measure the performance of models.
"""

import torch
import numpy as np
from tabulate import tabulate


class ConfusionMatrix():

  def __init__(self, nc):
    assert nc > 1
    self._nc = nc
    self.reset()


  def __call__(self, **kwargs):
    true = kwargs['true'].flatten()
    pred = kwargs['pred'].flatten()
    assert isinstance(true, np.ndarray) == True
    assert isinstance(pred, np.ndarray) == True
    assert np.all(true >= 0) == True and np.all(true < self._nc) == True
    assert np.all(pred >= 0) == True and np.all(pred < self._nc) == True
    cmatrix = np.bincount(self._nc * true + pred, minlength=self._nc**2) \
                  .reshape(self._nc, self._nc)
    self.cmatrix += cmatrix

  def mean_iu(self):
    cf = self.cmatrix
    iu = np.diag(cf) / (cf.sum(1) + cf.sum(0) - np.diag(cf))
    miu = np.nanmean(iu)
    return miu

  def pixel_accuracy(self):
    acc = np.diag(self.cmatrix).sum() / self.cmatrix.sum()
    per_class_acc = np.diag(self.cmatrix) / self.cmatrix.sum(1)
    return acc, per_class_acc

  def set_nc(self, nc):
    self._nc = nc
    self.reset()

  def reset(self):
    self.cmatrix = np.zeros((self._nc, self._nc))
    
    
    
def confusion_matrix(nc, **kwargs):
  """
  Creates a Confusion Matrix.

  Arguments
  ---------
  cls  : int
         The number of classes.

  true : np.ndarray
         The True Predictions.

  pred : np.ndarray
         The Predcited values.

  Returns
  -------
  cmatrix : np.ndarray
            The Confusion Matrix
  """
  # NOTE Using kwargs to ensure the user passes the `true` and `pred` arguments
  # correctly, as if allowed positional arguments then it wouldn't have been
  # posssible to assert whether the user is passing the ground truths for `true`
  # and predictions as `pred`. This will need the user to explicitly adopt
  # keyword arguments and thus enforcing the user to not make a mistake.
  true = kwargs['true']
  pred = kwargs['pred']
  assert isinstance(true, np.ndarray) == True
  assert isinstance(pred, np.ndarray) == True
  assert np.all(true >= 0) == True and np.all(true < nc) == True
  assert np.all(pred >= 0) == True and np.all(pred < nc) == True
  assert nc > 1
  cmatrix = np.bincount(nc * true.flatten() + pred.flatten(), minlength=nc**2)\
                .reshape(nc, nc)
  return cmatrix


def mean_iu(nc, **kwargs):
  """
  Calculate the mean Intersection over Union.
  """
  cmatrix = confusion_matrix(nc, **kwargs)
  iu = np.diag(cmatrix) / (cmatrix.sum(1) + cmatrix.sum(0) - np.diag(cmatrix))
  miu = np.nanmean(iu)
  return miu


def pixel_accuracy(nc, **kwargs):
  """
  Calculates the Pixel Accuracy.
  """
  cmatrix = confusion_matrix(nc, **kwargs)
  acc = np.diag(cmatrix).sum() / cmatrix.sum()
  per_class_acc = np.diag(cmatrix) / cmatrix.sum(1)
  return acc, per_class_acc


def accuracy(out:torch.Tensor, target:torch.Tensor, topk:tuple=(1,)) -> list:
  """
  Function to help in measuring the accuracy of the predicted labels
  against the true labels.

  Arguments
  ---------
  out : torch.Tensor, [N x C] where C is the number of classes
        The predicted logits 
  pred : torch.Tensor, [N,]
          Total number of elements.
  """ 
  
  with torch.no_grad():
    assert out.shape[0] == target.shape[0]
    assert max(target) <= out.shape[1]
    _, pred = out.topk(max(topk), 1, True, True); pred.t_()
    corr = pred.eq(target.view(1, -1).expand_as(pred))
    acc_list = []
    for k in topk:
      corr_k = corr[:k].sum().item()
      acc_list.append(corr_k / out.size(0))
    return acc_list
