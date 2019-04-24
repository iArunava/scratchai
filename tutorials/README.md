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
#### One Calls

1. Classify an image
```
>>> from scratchai.one_call import *
>>> classify('https://proservegroup.com/ekmps/shops/proservegroup/images/gilbert-revolution-x-size-5-match-rugby-ball-3859-p.jpg')
'rugby ball'
```

2. Perform a style Transfer
```
>>> from scratchai.one_call import *
>>> stransfer('https://proservegroup.com/ekmps/shops/proservegroup/images/gilbert-revolution-x-size-5-match-rugby-ball-3859-p.jpg', save=1)
```
Notes: Available Styles = ['elephant_skin']
