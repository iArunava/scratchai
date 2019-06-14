import torch.nn as nn
import torch

class PredictNeuron(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(1, 4)
  def forward(self, x):
    return self.linear(x)

class Neuron(nn.Module):
  def __init__(self, name='n'):
    super().__init__()
    self.param = nn.Linear(1, 1)
    self.route1 = nn.Parameter(torch.randn(1))
    self.route2 = nn.Parameter(torch.randn(1))
    self.route3 = nn.Parameter(torch.randn(1))
    self.route4 = nn.Parameter(torch.randn(1))
    self.pred = PredictNeuron()

    self.name = name

  def forward(self, x):
    #x_copy = x.detach().clone().requires_grad_(False)
    x = x.unsqueeze(1) if len(x.shape) == 1 else x
    #print (x.shape)
    #input()
    x = self.param(x)
    pred = torch.argmax(self.pred(x), dim=1)
    return x, pred

  def __name__(self):
    return self.name
    

class M(nn.Module):
  def __init__(self):
    super().__init__()

    self.v1 = Neuron('1')
    self.v2 = Neuron('2')
    self.v3 = Neuron('3')
    self.v4 = Neuron('4')
    self.v5 = Neuron('5')
  
  def forward(self, x):
    x, pred = self.v1(x)
    print (pred)
    for i in range(5):
      prev_neuron = int('1' if i == 0 else pred)
      next_neuron = ((prev_neuron + (pred + 1)) % 5).item()
      next_neuron = str(next_neuron + (1 if next_neuron == 0 else 0))
      print (next_neuron)
      x, pred = getattr(self, 'v'+next_neuron)(x)
    return x
      
    x = self.route(neuron, x)
    return x
"""
  def route(self, n, x):
    for _ in range(10):
      #x_copy = x.detach().clone().requires_grad_(False)
      pred = torch.argmax(n[-1](x), dim=1)
      self.forward(x, n[pred])
"""
