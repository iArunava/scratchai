from pathlib import Path


__all__ = ['version', 'home']


version = '0.0.1dev'
home = str(Path.home()) + '/.scratchai/'
IMGNET12 = 'imagenet12'
MNIST = 'mnist'
CIFAR10 = 'cifar10'
