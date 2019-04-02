import torch
import torch.nn as nn
import warnings
from tqdm import tqdm

class TrainObj(object):
    
    def __init__(self, lr, epochs, mt, device, se, dltr, dltt=None, fn='ckpt', test_run=False):
        '''
        A class to help in efficient training
        Arguments:
        :: lr - The learning rate to be used
        :: epochs - # of epochs
        :: mt - Model Type.
                Choices - ['clf', 'seg']
        :: device - 'cpu' || 'cuda'
        :: se - make a checkpoint every se epochs
        :: fn - the name to be used for saving the ckpt file
        :: test_run - If True, the train loop is completed with just one iteration
                      over the dataloaders, in order to check everything is working
                      as expected.
                      Default - False
        '''
        
        mtchoices = ['clf', 'seg']
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
        self.dltr = dltr
        self.dltt = dltt
        self.ipe = len(self.dltr)

    def train(self):
        print ('Hey whassup!')

    def _trainseg(self):
        '''
        Method to help in training Segmentation datasets

        Arguments;
        :: m
        '''
        for e in range(1, self.epochs+1):

            train_loss = 0
            model.train()
            for ii, data in enumerate(tqdm(self.dltr)): 

                img, lab = data

                if img.shape[0] == 1:
                    print (img.shape, ii, e)
                    continue
                    
                img, lab = img.to(self.device), lab.to(self.device)
                lab *= 255
                
                optimizer.zero_grad()

                out = model(img.float())
                
                loss = criterion(out, lab.squeeze(1).long())
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                
                if ii % show_every == 0:
                    out5 = show_cscpaes(model, H, W)
                    checkpoint = {
                        'epochs' : e,
                        'model_state_dict' : model.state_dict(),
                        'opt_state_dict' : optimizer.state_dict()
                    }
                    torch.save(checkpoint, './ckpt-dlabv3-{}-{:2f}.pth'.format(e, train_loss))
                    print ('Model saved!')
                    print ('Epoch {}/{}...'.format(e, epochs),
                        'Loss {:6f}'.format(loss.item()))
                
                #break

            print ()
            train_losses.append(train_loss)

            
            if (e+1) % print_every == 0:
                print ('Epoch {}/{}...'.format(e, epochs),
                        'Loss {:6f}'.format(train_loss))
            '''
            if e % eval_every == 0:
                with torch.no_grad():
                    model.eval()

                    eval_loss = 0

                    for _ in tqdm(range(bc_eval)):
                        inputs, labels = next(eval_pipe)

                        inputs, labels = inputs.to(device), labels.to(device)
                        out = model(inputs.float())

                        loss = criterion(out, labels.long())

                        eval_loss += loss.item()

                    print ()
                    print ('Loss {:6f}'.format(eval_loss))

                    eval_losses.append(eval_loss)
            
            '''
            scheduler.step(train_loss)
            
            '''
            if e % save_every == 0:
                
                
                show_pascal(model, training_path, all_tests[np.random.randint(0, len(all_tests))])
                checkpoint = {
                    'epochs' : e,
                    'state_dict' : model.state_dict(),
                    'opt_state_dict' : optimizer.state_dict()
                }
                torch.save(checkpoint, '/content/ckpt-enet-{}-{:2f}.pth'.format(e, train_loss))
                print ('Model saved!')
            '''
            
        #     show(model, all_tests[np.random.randint(0, len(all_tests))])
        #     show_pascal(model, training_path, all_tests[np.random.randint(0, len(all_tests))])


    def _trainclf(self, moc, dltr, dltt=None):
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
