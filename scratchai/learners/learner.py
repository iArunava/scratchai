import torch
import torch.nn as nn
from tabulate import tabulate
from .trainops.train import TrainObj
import torch.optim as optim
from tqdm import tqdm

__all__ = ['Learner', 'SegLearner']

    
class Learner(object):
    
    def __init__(self, net:nn.Module, loader=None, lr:float=1e-3, epochs:int=10, mt:str='seg', \
                 crit:nn.Module=nn.CrossEntropyLoss, opt=optim.Adam, she:int=5, device='cuda', metrics:list=None, \
                 sae:int=1, pe:int=1, wd:int=1e-4, trainiter:int=None, valiter:int=None):
        """
        A Learner Object

        Arguments:
            :: net - The model to train
            :: loader - The DataLoader from where to get the data
            :: epochs - The number of epochs
            :: she - Interval in which to checkpoint
            :: pe - Interval in which it prints the loss and other metrics
            :: wd - Weight Decay for L2 Regularization
        """

        self.lr = lr
        self.epochs = epochs
        self.mt = mt
        self.device = device
        self.net = net.to(self.device)
        self.she = she
        self.sae = sae
        self.pe = pe
        self.loader = loader
        self.metrics = metrics
        self.wd = wd
        self.trainiter = trainiter
        self.valiter = valiter

        self.tlosses = []
        self.vlosses = []

        '''
        self.dltr = self.loader.get_dltr()
        self.dltt = self.loader.get_dltt()
        self.trainiter = len(self.dltr) // self.dltr.batch_size if not trainiter \
                         else trainiter
        self.valiter = len(self.dltt) // self.dltt.batch_size if not valiter \
                         else valiter
        '''
        self.crit = crit()
        self.opt = opt(net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.h = 512
        self.w = 512

    def fit(self):
        assert self.dltr is not None
        if self.mt == 'seg':
            self._trainseg()

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
    
    ###################################################################################
    #################################################################################
    #########################Ongoing Print Functions################################
    #################################################################################
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


class SegLearner(Learner):
    def __init__(self, *args, **kwargs):
        """
        The Learner Object that helps train Segmentation Datasets

        Arguments:
            :: net - The model to train
            :: loader - The DataLoader from where to get the data
            :: epochs - The number of epochs
            :: she - Interval in which to checkpoint
            :: pe - Interval in which it prints the loss and other metrics
            :: wd - Weight Decay for L2 Regularization
        """
        
        super().__init__(*args, **kwargs)
        
        '''
        self.dltr = self.loader.get_dltr()
        self.dltt = self.loader.get_dltt()
        '''
        self.trainiter = self.loader.len if not self.trainiter \
                         else self.trainiter
        self.valiter = self.loader.len if not self.valiter \
                         else self.valiter

    def fit(self):
        self._trainseg()

    def _trainseg(self):
        """
        Method to help in training of Segmentation datasets
        """

        for e in range(1, self.epochs+1):
            trloss = 0
            self.net.train()

            for ii in tqdm(range(self.trainiter)):

                img, lab = next(iter(self.loader.get_batch()))

                # TODO hack to skip batches with size = 1
                if img.shape[0] == 1:
                    continue

                img, lab = img.to(self.device), lab.to(self.device)
                # TODO Introduce transforms in get_batch
                #lab *= 255

                self.opt.zero_grad()
                out = self.net(img.float())
                loss = self.crit(out, lab.long())
                loss.backward()
                self.opt.step()

                trloss += loss.item()
                break
                
            self.tlosses.append(trloss)

            if ii % self.she == 0:
                #show_camvid(self.net, self.h, self.w)
                print ('Epoch {}/{}...'.format(e, self.epochs),
                    'Loss {:6f}'.format(loss.item()))

            if e % self.sae == 0:
                checkpoint = {
                    'epochs' : e,
                    'model_state_dict' : self.net.state_dict(),
                    'opt_state_dict' : self.opt.state_dict()
                }
                torch.save(checkpoint, './ckpt-{:2f}.pth'.format(trloss))
                print ('Model saved!')
                
            if e % self.pe == 0:
                print ('Epoch {}/{}...'.format(e, self.epochs),
                        'Loss {:6f}'.format(trloss))

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
