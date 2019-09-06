"""
Linear GANs.
A Simple Linear implementation of GANs. (with Linear layers not Conv Layers)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F



__all__ = ['LinearD', 'LinearG', 'get_lineargan_mnist']


# TODO Make the implementation more General with DCGANs. Such that 
# If new layers are introduced then we can easily use the existing backbone.
# NOTE Not Working

class LinearD(nn.Module):
  def __init__(self, input_size=784, hidden_dim=128, output_size=1):
    '''
    The constructor class for the Discriminator

    Arguments:
    - input_size : the number of input neurons
    - hidden_dim : the number of hidden neurons in the last layer
    - output_size : the number of output neurons
    '''
    print ('[INFO] The net is designed with MNIST images in mind')
    super().__init__()

    # Define the class variables
    self.input_size = input_size
    self.hidden_dim = hidden_dim
    self.output_size = output_size

    # Define the required modules for this architecture
    self.linear1 = nn.Linear(self.input_size, self.hidden_dim*4)
    self.linear2 = nn.Linear(self.hidden_dim*4, self.hidden_dim*2)
    self.linear3 = nn.Linear(self.hidden_dim*2, self.hidden_dim)

    self.linear4 = nn.Linear(self.hidden_dim, self.output_size)

    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    '''
    This method defines the forward pass for the Discriminator module

    Arguments:
    - x : the input to the network

    Returns:
    - out : the output of the network
    '''
    # Flatten the images
    x = x.view(-1, 784)

    # Forward pass
    x = F.leaky_relu(self.linear1(x), 0.2)
    x = self.dropout(x)

    x = F.leaky_relu(self.linear2(x), 0.2)
    x = self.dropout(x)

    x = F.leaky_relu(self.linear3(x), 0.2)
    x = self.dropout(x)

    out = self.linear4(x)
    out = F.sigmoid(out)

    return out



class LinearG(nn.Module):
    
  def __init__(self, input_size=100, hidden_dim=32, output_size=784):
    '''
    This is the constructor class for the generator

    Arguments:
    - input_size : The hidden size of the vector of the latent sample
    - hidden_dim : The number of neurons for the last number of layers
    - output_size : The number neurons for the output layer
    '''
    print ('[INFO] The net is designed with MNIST images in mind')
    super().__init__()

    # Define the class variables
    self.input_size = input_size
    self.hidden_dim = hidden_dim
    self.output_size = output_size

    # Define the modules required by this class
    self.fc1 = nn.Linear(self.input_size, self.hidden_dim)
    self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim*2)
    self.fc3 = nn.Linear(self.hidden_dim*2, self.hidden_dim*4)

    self.fc4 = nn.Linear(hidden_dim*4, output_size)

    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    '''
    This method defines the forward pass for Generator module
    
    Arguments:
    - x : the input to the network

    Returns:
    - out : the output of the network
    '''

    x = F.leaky_relu(self.fc1(x), 0.2)
    x = self.dropout(x)

    x = F.leaky_relu(self.fc2(x), 0.2)
    x = self.dropout(x)

    x = F.leaky_relu(self.fc3(x), 0.2)
    x = self.dropout(x)

    x = self.fc4(x)
    out = F.tanh(x)

    out = out.reshape((-1, 1, 28, 28))

    return out


# ==============================================================
#
# Easy Calls
# 
# ==============================================================

def get_lineargan_mnist():
  return LinearG(), LinearD()
