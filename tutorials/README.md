### Tutorials

1. Train a net on MNIST
```
>>> from scratchai import *
>>> net = nets.lenet_mnist(pretrained=False)
>>> learners.quicktrain.mnist(net)
```

2. Train a net on CIFAR10
```
>>> from scratchai import *
>>> net = nets.lenet_cifar10(pretrained=False)
>>> learners.quicktrain.cifar10(net)
```

3. Train a UNet [WIP]
```
>>> import scratchai
>>> net = scratchai.nets.UNet(3, 32)
>>> load = scratchai.DataLoader.camvid('.', download=True)
>>> learner = scratchai.learners.Learner(net, load)
>>> learner.fit()
```

4. Train a ENet [WIP]
```
>>> import scratchai
>>> net = scratchai.nets.ENet(32)
>>> load = scratchai.DataLoader.camvid('.', download=True)
>>> learner = scratchai.learners.SegLearner(net, load)
>>> learner.fit()
```

#### One Calls

1. Classify an image (with Imagenet)
```
>>> from scratchai.one_call import *
>>> classify('https://proservegroup.com/ekmps/shops/proservegroup/images/gilbert-revolution-x-size-5-match-rugby-ball-3859-p.jpg')
'rugby ball'
```

2. Classify an image (with MNIST)
```
>>> from scratchai.one_call import *
>>> classify('http://rhappy.fun/blog/hello-keras/img/unnamed-chunk-22-2.png', nstr='lenet_mnist', trf='rz28_tt_normmnist')
'eight'
```

3. Perform a style Transfer
```
>>> from scratchai.one_call import *
>>> stransfer('https://proservegroup.com/ekmps/shops/proservegroup/images/gilbert-revolution-x-size-5-match-rugby-ball-3859-p.jpg', save=1)
```

### Image Utils
1. Thresholding Image using thresh_img
```
>>> from scratchai.imgutils import  *
>>> threshold_img = thresh_img(img,[50,50,50],tcol = [25,25,25])
```




Notes: Available Styles = ['elephant_skin']
