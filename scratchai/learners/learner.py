import torch
import torch.nn as nn
from tabulate import tabulate
from .trainops.train import TrainObj
import torch.optim as optim
from tqdm import tqdm

__all__ = ['Learner']

class Learner(object):
    
    def __init__(self, net:nn.Module, loader=None, dltt=None, lr:float=1e-3, epochs:int=10, mt:str='seg', \
                 crit:nn.Module=nn.CrossEntropyLoss, opt=optim.Adam, se:int=5, device='cuda', metrics:list=None, e:int=10):
        '''
        A Learner Object

        Arguments:
        :: model - The model to train
        :: loader - The DataLoader from where to get the data
        :: e - The number of epochs
        :: se - Interval in which to checkpoint
        '''

        self.lr = lr
        self.epochs = epochs
        self.mt = mt
        self.net = net
        self.se = se
        self.loader = loader
        self.dltr = self.loader.get_dltr()
        self.dltt = self.loader.get_dltt()
        self.metrics = metrics
        self.device = device
        self.e = e

        self.crit = crit()
        self.opt = opt(net.parameters(), lr=self.lr)

        self.h = 512
        self.w = 512

    def fit(self):
        assert self.dltr is not None
        if self.mt == 'seg':
            self._trainseg()

    def _trainseg(self):
        '''
        Method to help in training Segmentation datasets

        Arguments;
        :: m
        '''
        for e in range(1, self.epochs+1):

            train_loss = 0
            self.net.train()
            for ii, data in enumerate(tqdm(self.dltr)):

                img, lab = data

                if img.shape[0] == 1:
                    print (img.shape, ii, e)
                    continue
                    
                img, lab = img.to(self.device), lab.to(self.device)
                lab *= 255
                
                self.opt.zero_grad()

                out = self.net(img.float())
                
                loss = self.crit(out, lab.squeeze(1).long())
                loss.backward()
                self.opt.step()

                train_loss += loss.item()
                
                if ii % show_every == 0:
                    out5 = show_cscpaes(self.net, H, W)
                    checkpoint = {
                        'epochs' : e,
                        'model_state_dict' : self.net.state_dict(),
                        'opt_state_dict' : optimizer.state_dict()
                    }
                    torch.save(checkpoint, './ckpt-dlabv3-{}-{:2f}.pth'.format(e, train_loss))
                    print ('Model saved!')
                    print ('Epoch {}/{}...'.format(e, epochs),
                        'Loss {:6f}'.format(loss.item()))
                
            print ()
            train_losses.append(train_loss)
            
            if (e+1) % print_every == 0:
                print ('Epoch {}/{}...'.format(e, epochs),
                        'Loss {:6f}'.format(train_loss))
            '''
            if e % eval_every == 0:
                with torch.no_grad():
                    self.net.eval()

                    eval_loss = 0

                    for _ in tqdm(range(bc_eval)):
                        inputs, labels = next(eval_pipe)

                        inputs, labels = inputs.to(device), labels.to(device)
                        out = self.net(inputs.float())

                        loss = criterion(out, labels.long())

                        eval_loss += loss.item()

                    print ()
                    print ('Loss {:6f}'.format(eval_loss))

                    eval_losses.append(eval_loss)
            
            '''
            #scheduler.step(train_loss)
            
            '''
            if e % save_every == 0:
                
                
                show_pascal(self.net, training_path, all_tests[np.random.randint(0, len(all_tests))])
                checkpoint = {
                    'epochs' : e,
                    'state_dict' : self.net.state_dict(),
                    'opt_state_dict' : optimizer.state_dict()
                }
                torch.save(checkpoint, '/content/ckpt-enet-{}-{:2f}.pth'.format(e, train_loss))
                print ('Model saved!')
            '''
            
        #     show(self.net, all_tests[np.random.randint(0, len(all_tests))])
        #     show_pascal(self.net, training_path, all_tests[np.random.randint(0, len(all_tests))])


    def _trainclf(self, moc, dltr, dltt=None):
        '''
        Method to help in training.
        Arguments:
        :: moc - A list with the self.net to train,
                                     optimizer, criterion
                     Expected: [self.net, criterion, optimizer]
        :: dltr - A DataLoader based on the Training set
        :: dltt - A DataLoader based on the test/eval set
        '''
        
        assert isinstance(moc, list)
        trl = float('Inf')
        vrl = float('Inf')

        for e in range(self.epochs):
            
            trl = 0
            moc[0].train()

            for ii, data in enumerate(tqdm(dltr)):
                images, labels = data
                images, labels = images.to(self.d), labels.to(self.d)

                moc[2].zero_grad()

                out = moc[0](images)
                loss = moc[1](out, labels)
                loss.backward()
                
                moc[2].step()
                trl += loss.item()

                self.trloss.append(trl)

            else:
                if dltt is None:
                    continue

                with torch.no_grad():
                    moc[0].eval()

                    acc = 0
                    vrl = 0

                    for ii, data in enumerate(tqdm(dltt)):
                        images, labels = data
                        images, labels = images.to(self.d), labels.to(self.d)

                        out = moc[0](images)
                        loss = moc[1](out, labels)
                        _, preds = torch.max(out.data, 1)

                        #top_k, top_class = ps.topk(1, dim=1)

                        corr = (preds == labels).sum().item()

                        vrl += loss.item()
                        self.ttloss.append(vrl)
                
                if dltt is not None:
                    cacc = (corr / len(dltt)) * 100
                else:
                    cacc = (corr / len(dltr)) * 100
                    warnings.warn("Calculating Accuracy on the train set!")

                self.acclist.append(cacc)

                if e+1 % self.se == 0:
                    ckpt = {
                        'epoch' : e+1,
                        'model_state_dict' : moc[0].state_dict(),
                        'opt_state_dict' : moc[2].state_dict()
                    }
                    torch.save(ckpt, '{}-{:.2f}'.format(self.fn, cacc))

                print ('Epochs {}/{} || Train Loss: {:.2f} || ' \
                       'Test Loss: {} || Accuracy {:.2f}' \
                       .format(e+1, self.epochs, trl, vrl, cacc))

    def conv_out_size(self, net):
        kh, kw = net.kernel_size if type(net.kernel_size) == tuple else (net.kernel_size, net.kernel_size)
        sh, sw = net.stride if type(net.stride) == tuple else (net.stride, net.stride)
        ph, pw = net.padding if type(net.padding) == tuple else (net.padding, net.padding)

        self.h = (int) ((self.h - kh + (2 * ph)) / sh) + 1
        self.w = (int) ((self.w - kw + (2 * pw)) / sw) + 1
        return self.h, self.w

    def unet_eblock_out(self, net):
        self.h = (int) ((self.h * 2) - 4)
        self.w = (int) ((self.w * 2) - 4)
        return self.h, self.w
    
    def summary(self):
        layers = [['Input'], [(self.h, self.w)]]
        print (tabulate(layers))
        self._summary(self.net)

    def _summary(self, net):
        layers = []
        for m in net.children():
            temp = []
            if isinstance(m, nn.Sequential):
                self._summary(m)

            elif isinstance(m, nn.Conv2d):
                temp.append('Conv2d({}, {}, {})'.format(m.kernel_size, m.stride, m.padding))
                temp.append('{}'.format(self.conv_out_size(m)))

            elif str(m.__class__).split('.')[-1][:-2] == 'UNet_EBlock':
                temp.append('UNet_EBlock({}, {})'.format(m.uc.in_channels, m.uc.out_channels))
                temp.append('{}'.format(self.unet_eblock_out(m)))

            elif str(m.__class__).split('.')[-1][:-2] == 'MaxPool2d':
                temp.append('MaxPool2d({}, {}, {})'.format(m.kernel_size, m.stride, m.padding))
                temp.append('{}'.format(self.conv_out_size(m)))
            else:
                temp.append('ReLU')
                temp.append('{}'.format((self.h, self.w)))
            
            if len(temp) > 0:
                layers.append(temp)
        
        print (tabulate(layers))
