import torch
import torch.nn as nn

class SimpleSegModel(nn.Module):
  def __init__(self, backbone, aux, head, head_ic):
    super().__init__()
    self.backbone = backbone
    self.aux = aux
    self.head_ic = head_ic
    self.head = head

  def 
