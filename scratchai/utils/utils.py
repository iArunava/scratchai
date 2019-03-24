import torch
import torch.nn as nn

def count_modules(net:nn.Module):
    allm = []
    mdict = {}
    for m in net.modules():
        name = m.__class__.__name__
        if name in allm:
            mdict[name] += 1
        else:
            allm.append(name)
            mdict[name] = 1

    return mdict

'''
def count_parameters(net:nn.Module):
    for m in net.modules():
'''
