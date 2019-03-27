import torch
import torch.nn as nn
import warnings

class TrainObj(Object):
    
    def __init__(self, lr, epochs, mt, device, se, fn='ckpt'):
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
        '''
        
        mtchoices = ['classifier']
        dvchoices = ['cpu', 'cuda']
        assert mt in mtchoices
        assert device in dvchoices

        self.lr = lr
        self.epochs = epochs
        self.mt = mt
        self.d = device
        self.trloss = []
        self.ttloss = []
        self.acc = []

    def train(self, moc, dltr, dltt=None):
        '''
        Method to help in training.

        Arguments:
        :: moc - A list with the model to train,
                                     optimizer, criterion
                     Expected: [model, optimizer, criterion]
        :: dltr - A DataLoader based on the Training set
        :: dltt - A DataLoader based on the test/eval set
        '''
        
        assert isinstance(moc, list)
        trl = None
        vrl = None

        for e in tqdm(range(self.epochs)):
            
            trl = 0
            moc[0].train()

            for ii, data in enumerate(dltr):
                images, labels = data
                images, labels = images.to(self.d), labels.to(self.d)

                moc[2].zero_grad()

                log_ps = moc[0](images)
                loss = moc[1](log_ps, labels)
                loss.backward()
                
                moc[2].step()
                trl += loss.item()

                trloss.append(trl)

            else:
                if dltt is None:
                    continue

                with torch.no_grad():
                    moc[0].eval()

                    acc = 0
                    vrl = 0

                    for ii, data in enumerate(dltt):
                        images, labels = data
                        images, labels = images.to(self.d), labels.to(self.d)

                        out = moc[0](images)
                        loss = moc[1](log_ps, labels)
                        _, preds = torch.max(out.data, 1)

                        #top_k, top_class = ps.topk(1, dim=1)

                        equals = (preds == labels).sum().item()
                        acc += torch.mean(equals.type(torch.FloatTensor))

                        rvl += loss.item()
                        ttloss.append(vrl)
                
                if dltt is None:
                    cacc = (acc / len(dltt)) * 100
                else:
                    cacc = (acc / len(dltr)) * 100
                    warnings.warn("Calculating Accuracy on the train set!")

                acclist.append(cacc)

                if e+1 % se == 0:
                    ckpt = {
                        'epoch' : e+1,
                        'model_state_dict' : moc[0].state_dict()
                        'opt_state_dict' : moc[2].state_dict()
                    }
                    torch.save(ckpt, '{}-{:.2f}'.format(cacc)

                print ('Epochs {}/{} || Train Loss: {:.3f} \
                       Test Loss: {:.3f} || Accuracy {:.3f}' \
                       .format(e+1, self.epochs, trl, vrl, cacc))
