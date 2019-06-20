"""
The metrics used to measure the performance of models.
"""

import torch
import numpy as np
from tabulate import tabulate


# TODO Needs tests
def confusion_matrix(true, pred, nc):
  """
  Creates a Confusion Matrix.

  Arguments
  ---------
  true : np.ndarray
         The True Predictions.

  pred : np.ndarray
         The Predcited values.

  cls  : int
         The number of classes.

  Returns
  -------
  cmatrix : np.ndarray
            The Confusion Matrix
  """
  # NOTE 1st argument True Values and 2nd argument Pred values
  # NOTE All the calculations that are done based on the cmatrix returned from
  # this function is coded in a way such that it depends on the order of the
  # arguments passed in. 1st argument are the Ground Truths and the Second 
  # argument are the Predictions from the model. I am writing this big comment
  # just because this function won't throw an error if the order of the true
  # and preds are reversed but will fail and give erroneous results silently.
  assert isinstance(true, np.ndarray) == True
  assert isinstance(pred, np.ndarray) == True
  assert np.all(true >= 0) == True and np.all(true < nc) == True
  assert np.all(pred >= 0) == True and np.all(pred < nc) == True
  assert nc > 1
  cmatrix = np.bincount(nc * true.flatten() + pred.flatten(), minlength=nc**2)\
                .reshape(nc, nc)
  return cmatrix


def mean_iu(true, pred, nc):
  """
  Calculate the mean Intersection over Union.
  """
  cmatrix = confusion_matrix(true, pred, nc)
  iu = np.diag(cmatrix) / (cmatrix.sum(1) + cmatrix.sum(0) - np.diag(cmatrix))
  miu = np.nanmean(iu)
  return miu


def pixel_accuracy(true, pred, nc):
  """
  Calculates the Pixel Accuracy.
  """
  cmatrix = confusion_matrix(true, pred, nc)
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
