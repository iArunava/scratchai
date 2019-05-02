"""
Contains functions that helps in weight initialization.
"""

import torch
import torch.nn as nn
import numpy as np

def xavier_normal(m:nn.Module):
  """
  Xavier Normal Initialization to all the conv layers
  And the weight of batch norm is initialized to 1
  and the bias of the batch norm to 0

  Arguments
  ---------
  m : nn.Module
        The net which to init.
  """

  if isinstance(m, nn.Conv2d):
    nn.init.xavier_normal_(m.weight)
    if m.bias is not None:
      nn.init.zeros_(m.bias)
  # Add other norms (like nn.GroupNorm2d)
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)


def xavier_uniform(m:nn.Module):
  """
  Xavier Uniform Initialization to all the conv layers
  And the weight of batch norm is initialized to 1
  and the bias of the batch norm to 0

  Arguments
  ---------
  m : nn.Module
        The net which to init.
  """

  if isinstance(m, nn.Conv2d):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
      nn.init.zeros_(m.bias)
  # Add other norms (like nn.GroupNorm2d)
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)


def kaiming_normal(m:nn.Module):
  """
  Kaiming Normal Initialization to all the conv layers
  And the weight of batch norm is initialized to 1
  and the bias of the batch norm to 0

  Arguments
  ---------
  m : nn.Module
        The net which to init.
  """

  if isinstance(m, nn.Conv2d):
    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if m.bias is not None:
      nn.init.zeros_(m.bias)
  # Add other norms (like nn.GroupNorm2d)
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)


def kaiming_uniform(m:nn.Module):
  """
  Kaiming Uniform Initialization to all the conv layers
  And the weight of batch norm is initialized to 1
  and the bias of the batch norm to 0

  Arguments
  ---------
  m : nn.Module
        The net which to init.
  """

  if isinstance(m, nn.Conv2d):
    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
    if m.bias is not None:
      nn.init.zeros_(m.bias)
  elif isinstance(m, nn.Linear):
    nn.init.zeros_(m.bias)
  # Add other norms (like nn.GroupNorm2d)
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)


def msr_init(m:nn.Module):
  """
  MSR Initialization to all the conv layers
  And the weight of batch norm is initialized to 1
  and the bias of the batch norm to 0

  Arguments
  ---------
  m : nn.Module
        The net which to init.
  """

  if isinstance(m, nn.Conv2d):
    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
    nn.init.normal_(m.weight, mean=0, std=np.sqrt(2/n))
    if m.bias is not None:
      nn.init.zeros_(m.bias)
  elif isinstance(m, nn.Linear):
    nn.init.zeros_(m.bias)
  # Add other norms (like nn.GroupNorm2d)
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)
