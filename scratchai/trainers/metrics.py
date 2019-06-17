"""
The metrics used to measure the performance of models.
"""

import torch
import numpy as np
from tabulate import tabulate


def miou(pred, gt, nc, c2n=None):
  """
  Mean Intersection over Union (mIOU).

  Arguments
  --------
  pred : torch.tensor, [N x 3 x H x W]
        The original input images to the model.
  gt : torch.tensor, [N x H x W]
        The corresponding labels of the images.
  nc : int
       The number of classes
  c2n : dict
        The mapping from class idx to class names.

  Returns
  -------
  miou : float
         The mean intersection over union value.
  
  Notes
  -----
  Do note: if batches of data are passed it is necessary that
  inp - [N x H x W]
  lab - [N x H x W]

  where each matrix in lab is have each pixel value in the range of [0, C)
  where C is the number of classes.
  """
  # TODO This function needs testing

  assert len(list(pred.shape)) in [3, 2]
  # Assert batch_size, height and width matches
  assert pred.shape == gt.shape
  
  with torch.no_grad():
    # Convert torch.tensor to np.ndarray
    if isinstance(pred, torch.Tensor):
      pred = pred.clone().detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
      gt = gt.clone().detach().cpu().numpy()

    iou = {}
    miou = 0
    for cls in range(nc):
      inter = np.logical_and(pred == cls,  gt == cls).sum()
      union = np.logical_or(pred == cls, gt == cls).sum()
      iou[cls] = inter / union if inter != 0 and union != 0 else 1.
      miou += iou[cls]
  
  # TODO print the iou in table format
  #print (tabulate([list(iou.keys()), list(iou.values())]))

  return miou/nc


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
