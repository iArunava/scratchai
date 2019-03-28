# Imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from torchvision import datasets, transforms
import itertools
import matplotlib.gridspec as gridspec
from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output
import pydot

def bnconv(in_channels, out_channels, kernel_size, stride, padding, init=nn.init.kaiming_normal_,
           eps=1e-5, momentum=0.1, conv_first=True, relu='relu', nslope=0.2, norm='instance'):
    
    norm = nn.BatchNorm2d if norm == 'batch' else nn.InstanceNorm2d
    acti = nn.ReLU(inplace=True) if relu == 'relu' else nn.LeakyReLU(negative_slope=nslope, inplace=True)
    bias = norm == nn.InstanceNorm2d
    
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                     padding=padding, bias=bias)
    if init:
        init(conv.weight)
        if hasattr(conv, 'bias') and hasattr(conv.bias, 'data'): conv.bias.data.fill_(0.)
            
    layers = [conv, norm(out_channels, eps=eps, momentum=momentum), acti]
    
    return layers if relu is not None else layers[:2]


def bnconvt(in_channels, out_channels, kernel_size, stride, padding, init=nn.init.kaiming_normal_,
            eps=1e-5, momentum=0.1, conv_first=True, relu='relu', nslope=0.2, norm='instance'):

    norm = nn.BatchNorm2d if norm == 'batch' else nn.InstanceNorm2d
    acti = nn.ReLU(inplace=True) if relu == 'relu' else nn.LeakyReLU(negative_slope=nslope, inplace=True)
    bias = norm == nn.InstanceNorm2d
    
    conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=1, output_padding=1, bias=bias)
    
    if init:
        init(conv.weight)
        if hasattr(conv, 'bias') and hasattr(conv.bias, 'data'): conv.bias.data.fill_(0.)
        
    layers = [conv, norm(out_channels, eps=eps, momentum=momentum), acti]
    
    return layers if relu is not None else layers[:2]


class ResidualBlock2L(nn.Module):
  
    def __init__(self, ic_conv, oc_conv, expansion=4, stride=2, downsample=False,
                     conv_first=True, relu_after_add=True, norm='instance',
                     pad_type='reflect', eps=1e-5, momentum=0.1):
        '''
        This class defines the Residual Basic Block with 2 Conv Layers

        Arguments;
        - ic_conv : # of input channels
        - oc_conv : # of output channels for the final conv layers
        - downsample : Whether this block is to downsample the input
        - expansion : The expansion for the channels
                      Default: 4
        - stride : if downsample is True then the specified stride is used.
                   Default: 2
        - conv_first : Whether to apply conv before bn or otherwise.
                       Default: True

        - relu_after_add : Whether to apply the relu activation after
                           adding both the main and side branch.
                           Default: True
        '''

        super(ResidualBlock2L, self).__init__()

        assert(downsample == True or downsample == False)
        assert(relu_after_add == True or relu_after_add == False)
        self.downsample = downsample
        self.expansion = expansion
        self.relu_after_add = relu_after_add
        oc_convi = oc_conv // self.expansion

        stride = stride if self.downsample else 1

        layers = []

        p = 0
        if pad_type == 'reflect':
            layers += [nn.ReflectionPad2d(1)]
        elif pad_type == 'replicate':
            layers += [nn.ReplicationPad2d(1)]
        elif pad_type == 'pad':
            p = 1
        else:
            raise NotImplemented('This pad type is not implemented!')

        layers += [*bnconv(in_channels=ic_conv,
                                out_channels=oc_convi,
                                kernel_size=3,
                                padding=p,
                                stride=stride,
                                #eps=2e-5,
                                #momentum=0.9,
                                conv_first=True,
                                norm=norm,
                                eps=eps,
                                momentum=momentum)]

        p = 0
        if pad_type == 'reflect':
            layers += [nn.ReflectionPad2d(1)]
        elif pad_type == 'replicate':
            layers += [nn.ReplicationPad2d(1)]
        elif pad_type == 'pad':
            p = 1
        else:
            raise NotImplemented('This pad type is not implemented!')

        layers += [*bnconv(in_channels=oc_convi,
                            out_channels=oc_conv,
                            kernel_size=3,
                            padding=p,
                            stride=1,
                            #eps=2e-5,
                            #momentum=0.9,
                            conv_first=True,
                            relu=None,
                            norm=norm,
                            eps=eps,
                            momentum=momentum)
                   ]


        self.side = nn.Sequential(*layers)
    
    def forward(self, x): return x + self.side(x)

class CycleGAN_G(nn.Module):

    def __init__(self, ic_conv=64, norm='instance', pad_type='reflect', 
                 eps=1e-5, momentum=0.1):
        '''
        This class defines the network used for CycleGAN.
        This network is solely taken from the Supplementary material provided
        along with the Perceptual Losses paper by Johnson et al.
        '''
        
        super(CycleGAN, self).__init__()
        
        self.net = nn.Sequential(
                        nn.ReflectionPad2d(3),
                        
                        *bnconv(3, ic_conv, kernel_size=7, padding=0, stride=1,
                                conv_first=True, norm=norm, eps=eps, momentum=momentum),

                        *bnconv(ic_conv, ic_conv, kernel_size=3, padding=1, stride=2,
                                conv_first=True, norm=norm, eps=eps, momentum=momentum),

                        *bnconv(ic_conv, ic_conv*2, kernel_size=3, padding=1, stride=2,
                                conv_first=True, norm=norm, eps=eps, momentum=momentum),

                        *bnconv(ic_conv*2, ic_conv*4, kernel_size=3, padding=1, stride=2,
                                conv_first=True, norm=norm, eps=eps, momentum=momentum),

                        ResidualBlock2L(ic_conv*4, ic_conv*4,
                                        conv_first=True,
                                        relu_after_add=False,
                                        norm=norm,
                                        pad_type=pad_type,
                                        eps=eps,
                                        momentum=momentum),

                        ResidualBlock2L(ic_conv*4, ic_conv*4,
                                        conv_first=True,
                                        relu_after_add=False,
                                        norm=norm,
                                        pad_type=pad_type,
                                        eps=eps,
                                        momentum=momentum),

                        ResidualBlock2L(ic_conv*4, ic_conv*4,
                                        conv_first=True,
                                        relu_after_add=False,
                                        norm=norm,
                                        pad_type=pad_type,
                                        eps=eps,
                                        momentum=momentum),

                        ResidualBlock2L(ic_conv*4, ic_conv*4,
                                        conv_first=True,
                                        relu_after_add=False,
                                        norm=norm,
                                        pad_type=pad_type,
                                        eps=eps,
                                        momentum=momentum),

                        ResidualBlock2L(ic_conv*4, ic_conv*4,
                                        conv_first=True,
                                        relu_after_add=False,
                                        norm=norm,
                                        pad_type=pad_type,
                                        eps=eps,
                                        momentum=momentum),
                        
                        ResidualBlock2L(ic_conv*4, oc_conv=ic_conv*4,
                                        conv_first=True,
                                        relu_after_add=False,
                                        norm=norm,
                                        pad_type=pad_type,
                                        eps=eps,
                                        momentum=momentum),

                        *bnconvt(ic_conv*4, ic_conv*2, kernel_size=3, padding=1, stride=2,
                                conv_first=True, norm=norm, eps=eps, momentum=momentum),

                        *bnconvt(ic_conv*2, ic_conv, kernel_size=3, padding=1, stride=2,
                                conv_first=True, norm=norm, eps=eps, momentum=momentum),

                        *bnconvt(ic_conv, ic_conv, kernel_size=3, padding=1, stride=2,
                                conv_first=True, relu=False, norm=norm, eps=eps, momentum=momentum),
                        
                        nn.ReflectionPad2d(3), nn.Conv2d(ic_conv,3, kernel_size=7, padding=0, stride=1),
                        nn.Tanh()
            )

        self.reset_params()

    def forward(self, x): return self.net(x)
    
    def reset_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight.data, mean=1.0, std=0.02)
                nn.init.constant_(layer.bias.data, 0)
                
class CycleGAN_D(nn.Module):

    def __init__(self, iconv=64, num_conv=4):
        
        super(Discriminator, self).__init__()
        
        layers = [nn.Conv2d(3, iconv, kernel_size=4, stride=2, padding=1), 
                  nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        for _ in range(num_conv-2):
            layers += bnconv(iconv, iconv*2, kernel_size=4, stride=2, padding=1,
                                  relu='leaky', nslope=0.2)
            iconv *= 2

        layers += [*bnconv(iconv, iconv*2, kernel_size=4, stride=1, padding=1, relu='leaky', nslope=0.2),
                   nn.Conv2d(iconv*2, 1, kernel_size=4, stride=1, padding=1)]
        
        self.net = nn.Sequential(*layers)
        #self.reset_params()
        
    def forward(self, x): return self.net(x)
    
    def reset_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight.data, mean=1.0, std=0.02)
                nn.init.constant_(layer.bias.data, 0)
                
    def set_requires_grad(self, trfalse):
        for layer in self.parameters():
            layer.requires_grad = trfalse

########################################################################
#################### Image Pool ########################################
import torch
import torch.nn as nn
import numpy as np

class ImagePool(object):
    '''
    This is the class that stores the Pool of Images
    for the updates to make on the Discriminator.

    Arguments:
    - pool_size - The number of images to keep in buffer,
                  And to sample either from the pool or the
                  currently genererated images with a probability 
                  of 0.5
    '''

    def __init__(self, pool_size, device='cpu'):
        
        self.pool_size = pool_size
        self.device = device
        self.pool = []

    def sample(self, images):
        '''
        This function samples the images from the pool
        '''
        
        to_return = []
        for image in images:
            if len(self.pool) < self.pool_size:
                self.pool.append(image.cpu())
                to_return.append(image.unsqueeze(0))
            
            else:
                # p1 is the probability to should the current image
                # to add to return stack or the images in the pool
                p1 = np.random.rand()

                # p2 is probability with which the current image
                # is added to the pool. in a random idx
                p2 = np.random.rand()

                # If the p < 0.5
                # Sample from existing images
                if p1 < 0.5:
                    ridx = np.random.randint(0, self.pool_size)
                    to_return.append(self.pool[ridx].to(device).unsqueeze(0))
                else:
                    to_return.append(image.unsqueeze(0))

                # If the p < 0.6
                # Add the current image to pool
                if p2 < 0.6:
                    ridx = np.random.randint(0, self.pool_size)
                    self.pool[ridx] = image
        
        to_return = torch.cat(to_return, dim=0)
        return to_return


######################################################################
############# Show Results ###########################################
######################################################################

def show(g_x2y, g_y2x, xl, yl):
    x_data, _ = next(iter(xl))
    y_data, _ = next(iter(yl))
    
    with torch.no_grad():
        pred_y = g_x2y(x_data.to(device))
        pred_x = g_y2x(y_data.to(device))
    
    # Prep for showing
    x_data = x_data[0].squeeze(0).transpose(0, 1).transpose(1, 2).detach().cpu().numpy()
    y_data = y_data[0].squeeze(0).transpose(0, 1).transpose(1, 2).detach().cpu().numpy()
    pred_y = pred_y[0].squeeze(0).detach().transpose(0, 1).transpose(1, 2).cpu().numpy()
    pred_x = pred_x[0].squeeze(0).detach().transpose(0, 1).transpose(1, 2).cpu().numpy()
    to_show = [x_data, pred_y, y_data, pred_x]

    # Now show
    plt.figure(figsize = (10, 10))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.025, hspace=0.05)

    for i in range(4):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.imshow(unnormalize(to_show[i]))
    
    plt.show()
    
    
def unnormalize(img):
    #return (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    return (img * [0.5, 0.5, 0.5]) + [0.5, 0.5, 0.5]
    #return img

############################################################
############################################################
############################################################

# NOTE! Not Tested
class CycleGAN(object):
    
    def __init__(self, x_path, y_path, bs=4, transform=None, device='cuda', l_type='mse' betas=(0.5, 0.999), 
                 dlr=0.0002, glr=0.0002, lambA=10, lambB = 10, idt_loss=True, lamb_idt=0.1):
        
        super().__init__()
        
        # Store some vars
        self.d = device
        self.lambA = lambA
        self.lambB = lambB
        self.lamb_idt = lamb_idt
        self.idt_loss = idt_loss

        # Create the models
        self.gx2y = CycleGAN_G().to(self.d)
        self.gy2x = CycleGAN_G().to(self.d)
        self.dx = CycleGAN_D().to(self.d)
        self.dy = CycleGAN_D().to(self.d)
        
        # Create the Dataloaders
        x_data = datasets.ImageFolder(x_path, transform=transform)
        self.xloader = dataloader.DataLoader(x_data, batch_size=bs, shuffle=True)
        y_data = datasets.ImageFolder(y_path, transform=transform)
        self.yloader = dataloader.DataLoader(y_data, batch_size=bs, shuffle=True)

        # Define the criterion
        if l_type == 'mse':
            self.criterion = nn.MSELoss()
        elif l_type == 'l1':
            self.criterion = nn.L1Loss()
            print ('l1')
        self.cycle_c = nn.L1Loss()

        # Define the optimizer
        self.opt_G = optim.Adam(itertools.chain(self.gx2y.parameters(), self.gy2x.parameters()), lr=dlr, betas=betas)
        self.opt_D = optim.Adam(itertools.chain(self.dy.parameters(), self.dx.parameters()), lr=glr, betas=betas)

        # Intialize vars
        self.idt_x = 0
        self.idt_y = 0
    
    def crit(self, out, tar):
        target = torch.ones_like(out) if tar else torch.zeros_like(out)
        return self.criterion(real, target)

    def forward(self):
        self.fakeY = self.gx2y(self.realX)
        self.recnX = self.gy2x(self.fakeY)
        self.fakeX = self.gy2x(self.realY)
        self.recnY = self.gx2y(self.fakeX)

    def backward_G(self):
        self.gly = self.crit(self.fakeY, True)
        self.glx = self.crit(self.fakeX, True)

        self.cx = self.cycle_c(self.recnX, self.realX) * self.lambA
        self.cy = self.cycle_c(self.recnY, self.realY) * self.lambB

        if self.idt_loss:
            self.idt_x = self.criterion(self.gx2y(self.realY), self.realY) * self.lambA * self.lamb_idt
            self.idt_y = self.criterion(self.gy2x(self.realX), self.realX) * self.lambB * self.lamb_idt

        self.g_loss = self.cx + self.cy + self.gly + self.glx + self.idt_x + self.idt_y
        self.g_loss.backward()
    
    def backward_D(self):
        self.loss_fy = self.crit(self.dy(self.fakeY.detach()), False)
        self.loss_fx = self.crit(self.dx(self.fakeX.detach()), False)

        self.loss_ry = self.crit(self.dy(self.realY), True)
        self.loss_rx = self.crit(self.dx(self.realX), True)

        self.dx_loss = (loss_fx + loss_rx) * 0.5
        self.dx_loss.backward()

        self.dy_loss = (loss_fy + loss_fy) * 0.5
        self.dy_loss.backward()

    def do_one_iter(self):
        self.realX, _ = next(iter(self.xloader))
        self.realY, _ = next(iter(self.yloader))
        self.realX, self.realY = self.realX.to(self.d), self.realY.to(self.d)

        self.forward()

        self.dx.set_requires_grad(False); self.dy.set_requires_grad(False)
        self.gx2y.zero_grad(); self.gy2x.zero_grad()
        self.backward_G()
        self.opt_G.step()

        self.dx.set_requires_grad(True); self.dy.set_requires_grad(True)
        self.dx.zero_grad(); self.dy.zero_grad()
        self.backward_D()
        self.opt_D.step()
