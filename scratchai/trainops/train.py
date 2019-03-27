import torch
import torch.nn as nn
import warnings

class TrainObj(object):
    
    def __init__(self, lr, epochs, mt, device, se, fn='ckpt', test_run=False):
        '''
        A class to help in efficient training
        Arguments:
        :: lr - The learning rate to be used
        :: epochs - # of epochs
        :: mt - Model Type.
                Choices - ['classifier']
        :: device - 'cpu' || 'cuda'
        :: se - make a checkpoint every se epochs
        :: fn - the name to be used for saving the ckpt file
        :: test_run - If True, the train loop is completed with just one iteration
                      over the dataloaders, in order to check everything is working
                      as expected.
                      Default - False
        '''
        
        mtchoices = ['classifier']
        dvchoices = ['cpu', 'cuda']
        assert mt in mtchoices
        assert device in dvchoices

        self.lr = lr
        self.epochs = epochs
        self.mt = mt
        self.d = device
        self.se = se
        self.trloss = []
        self.ttloss = []
        self.acclist = []

    def train(self, moc, dltr, dltt=None):
        '''
        Method to help in training.
        Arguments:
        :: moc - A list with the model to train,
                                     optimizer, criterion
                     Expected: [model, criterion, optimizer]
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
