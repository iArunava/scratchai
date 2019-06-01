from scratchai import *
import torch.nn as nn

class M(nn.Module):
  def __init__(self):
    super().__init__()

    self.v1 = [nn.Linear(1, 1),
        nn.Parameter(torch.randn(1)),
        nn.Parameter(torch.randn(1)),
        nn.Parameter(torch.randn(1)),
        nn.Parameter(torch.randn(1))]

    self.v2 = [nn.Linear(1, 1),
      nn.Parameter(torch.randn(1)),
      nn.Parameter(torch.randn(1)),
      nn.Parameter(torch.randn(1)),
      nn.Parameter(torch.randn(1))]

    self.v3 = [nn.Linear(1, 1),
    self.c1v3 = nn.Parameter(torch.randn(1)),
    self.c2v3 = nn.Parameter(torch.randn(1)),
    self.c3v3 = nn.Parameter(torch.randn(1)),
    self.c4v3 = nn.Parameter(torch.randn(1))]

    self.v4 = [nn.Linear(1, 1)
    self.c1v4 = nn.Parameter(torch.randn(1)),
    self.c2v4 = nn.Parameter(torch.randn(1)),
    self.c3v4 = nn.Parameter(torch.randn(1)),
    self.c4v4 = nn.Parameter(torch.randn(1))]

    self.v5 = [nn.Linear(1, 1)
    self.c1v5 = nn.Parameter(torch.randn(1)),
    self.c2v5 = nn.Parameter(torch.randn(1)),
    self.c3v5 = nn.Parameter(torch.randn(1)),
    self.c4v5 = nn.Parameter(torch.randn(1))]
  
  def route(self, n, x):
    if x.item() < .25: x *= self.[1]
    elif x.item() < .5: x *= self.[2]
    elif x.item() < .75: x *= self.[3]
    else: x *= self.[4]
    
  def forward(self, x):
    x = self.v1[0](x)
    x = route(self.v1, x)
