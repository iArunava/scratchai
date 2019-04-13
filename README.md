# scratchai

## Builds

[![CircleCI](https://circleci.com/gh/iArunava/scratchai.svg?style=svg)](https://circleci.com/gh/iArunava/scratchai)

## Documentation

Table of Contents:
1. Classification
  - [Resnet]()
  - [Resnext]()
2. Segmentation
  - [UNet](https://github.com/iArunava/scratchai/blob/master/scratchai/nets/seg/unet.py)
  - [ENet](https://github.com/iArunava/scratchai/blob/master/scratchai/nets/seg/enet.py)
3. Generative Adversarial Networks
  - [DCGAN](https://github.com/iArunava/scratchai/blob/master/scratchai/nets/resnet.py)
  - [CycleGAN](https://github.com/iArunava/scratchai/blob/master/scratchai/nets/gans/cycle_gan.py)
  
  
### Tutorials

1. Train a UNet [WIP]
```
>>> import scratchai
>>> net = scratchai.nets.UNet(3, 32)
>>> load = scratchai.DataLoader.camvid('.', download=True)
>>> learner = scratchai.learners.Learner(net, load)
>>> learner.fit()
```
2. Train a ENet [WIP]
```
>>> import scratchai
>>> net = scratchai.nets.ENet(32)
>>> load = scratchai.DataLoader.camvid('.', download=True)
>>> learner = scratchai.learners.SegLearner(net, load)
>>> learner.fit()
```
