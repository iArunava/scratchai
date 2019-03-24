from .blocks import resblock, bnconv

class Resnet(nn.Module):
  
  def __init__(self, nc:int, block:nn.Module, layers:list, lconv:int=2, expansion:int=1, 
                     s1_channels:int=64, conv_first=True, inplace=True):
    '''
    The class that defines the ResNet module
    
    Arguments:
    - nc :: # of classes
    - s1_channels : # of channels for the output of the first stage
    - layers
    - lconv : # of conv layers in each Residual Block
    '''
    super(Resnet, self).__init__()
    
    layers = [bnconv(in_channels=3,
                out_channels=s1_channels,
                kernel_size=7,
                padding=3,
                stride=2,
                conv_first=conv_first),
    
              nn.MaxPool2d(kernel_size=3,
                                stride=2,
                                padding=1),
    
              resblock(block, inc=s1_channels,
                              outc=s1_channels*expansion,
                              num_layers=layers[0],
                              stride=1, conv_first=conv_first,
                              inplace=inplace),
    
              resblock(block, inc=s1_channels*expansion,
                              outc=s1_channels*expansion*2,
                              num_layers=layers[1], 
                              conv_first=conv_first,
                              inplace=inplace),
    
              resblock(block, inc=s1_channels*expansion*2,
                              outc=s1_channels*expansion*4,
                              num_layers=layers[2], 
                              conv_first=conv_first,
                              inplace=inplace),
    
              resblock(block, inc=s1_channels*expansion*4,
                              outc=s1_channels*expansion*8,
                              num_layers=layers[3], 
                              conv_first=conv_first,
                              inplace=inplace),
            ]
    
    self.net = nn.Sequential(*layers)
    
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    self.fc = nn.Linear(512, nc)
    
  def forward(self, x):
    
    bs = x.size(0)
    x = self.net(x)
    x = self.avgpool(x) if self.apool else x
    x = x.view(bs, -1)
    x = self.fc(x)
    return x
