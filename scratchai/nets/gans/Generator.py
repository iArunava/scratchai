import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    
    def __init__(self, input_size, hidden_dim, output_size):
        '''
        This is the constructor class for the generator

        Arguments:
        - input_size : The hidden size of the vector of the latent sample
        - hidden_dim : The number of neurons for the last number of layers
        - output_size : The number neurons for the output layer
        '''
        
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

        return out
