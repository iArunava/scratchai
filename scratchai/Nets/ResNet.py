from .blocks import resblock, bnconv

class ResNet(nn.Module):
  
  def __init__(self, nc:int, block:nn.Module, layers:list, lconv:int=2, expansion:int=1, 
                     s1_channels:int=64, apool:bool=True, conv_first=True, inplace=False):
    '''
    The class that defines the ResNet module
    
    Arguments:
    - nc :: # of classes
    - s1_channels : # of channels for the output of the first stage
    - layers
    - lconv : # of conv layers in each Residual Block
    '''
    super(ResNet, self).__init__()
    
    self.apool = apool
    
    layers = [BNConv(in_channels=3,
                out_channels=s1_channels,
                kernel_size=7,
                padding=3,
                stride=2,
                eps=2e-5,
                momentum=0.9,
                conv_first=True),
    
              nn.MaxPool2d(kernel_size=3,
                                stride=2),
    
              ResNetBlock(block, inc=s1_channels,
                              outc=s1_channels*expansion,
                              num_layers=layers[0],
                              stride=1, conv_first=conv_first,
                              inplace=inplace),
    
              ResNetBlock(block, inc=s1_channels*expansion,
                              outc=s1_channels*expansion*2,
                              num_layers=layers[1], 
                              conv_first=conv_first,
                              inplace=inplace),
    
              ResNetBlock(block, inc=s1_channels*expansion*2,
                              outc=s1_channels*expansion*4,
                              num_layers=layers[2], 
                              conv_first=conv_first,
                              inplace=inplace),
    
              ResNetBlock(block, inc=s1_channels*expansion*4,
                              outc=s1_channels*expansion*8,
                              num_layers=layers[3], 
                              conv_first=conv_first,
                              inplace=inplace),
            ]
    
    self.net = nn.Sequential(*layers)
    
    if self.apool:
      self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
    
    self.fc = nn.Linear(2048, nc)
    
    #'''
    for module in self.modules():
      #if isinstance(module, nn.Conv2d):
        #nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        #nn.init.xavier_normal_(module.weight, gain=0.02)
      if isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight.data, 1)
        #nn.init.constant_(module.bias, 0)
    #'''
    
  def forward(self, x):
    
    bs = x.size(0)
    x = self.net(x)
    x = self.avgpool(x) if self.apool else x
    x = x.view(bs, -1)
    x = self.fc(x)
    return x
