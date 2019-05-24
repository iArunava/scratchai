# scratchai

## Builds

[![CircleCI](https://circleci.com/gh/iArunava/scratchai.svg?style=svg)](https://circleci.com/gh/iArunava/scratchai)

## Documentation

Table of Contents:

1. Classification

| Model | Paper | Implementation | Configurations |
| :--- | :-----: | :--: | :--: |
| Lenet | http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf | [Implementation](https://github.com/iArunava/scratchai/blob/master/scratchai/nets/clf/lenet.py) | |
| Alexnet | https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf | [Implementation](https://github.com/iArunava/scratchai/blob/master/scratchai/nets/clf/alexnet.py) | |
| VGG | https://arxiv.org/pdf/1409.1556.pdf | [Implementation](https://github.com/iArunava/scratchai/blob/master/scratchai/nets/clf/vgg.py) | VGG11, VGG11_BN, VGG13, VGG13_BN, VGG16_BN, VGG19, VGG19_BN |
| Resnet | https://arxiv.org/abs/1512.03385 | [Implementation](https://github.com/iArunava/scratchai/blob/master/scratchai/nets/clf/resnet.py#L117) | |
| Resnext | https://arxiv.org/abs/1611.05431 | NA | |

2. Segmentation

| Model | Paper | Implementation |
| :--- | :-----: | :--: |
| UNet | https://arxiv.org/abs/1505.04597 | [Implementation](https://github.com/iArunava/scratchai/blob/master/scratchai/nets/seg/unet.py#L38) [Not checked] |
| ENet | https://arxiv.org/abs/1606.02147 | [Implementation](https://github.com/iArunava/scratchai/blob/master/scratchai/nets/seg/enet.py#L155) [Not checked] |

3. Generative Adversarial Networks

| Model | Paper | Implementation |
| :--- | :-----: | :--: |
| DCGAN | https://arxiv.org/abs/1511.06434 | NA |
| CycleGAN | https://arxiv.org/abs/1703.10593 | [Implementation](https://github.com/iArunava/scratchai/blob/master/scratchai/nets/gans/cycle_gan.py) [Not checked] |

4. Style Transfer

| Model | Paper | Implementation |
| :--- | :-----: | :--: |
| Image Transformation Network Justin et al. | [Perceptual Losses Paper](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)<br/>[Supplementary Material](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf) | [Implementation](https://github.com/iArunava/scratchai/blob/86d5011394592bde57eda40ba4682c8f26863b13/scratchai/nets/style_transfer/image_transformation_net.py#L75)

5. Attacks

| Attacks | Paper | Implementation |
| :--- | :-----: | :--: |
| Noise | NA | [Implementation](https://github.com/iArunava/scratchai/blob/master/scratchai/attacks/attacks/noise.py) |
| Semantic | https://arxiv.org/abs/1703.06857 | [Implementation](https://github.com/iArunava/scratchai/blob/master/scratchai/attacks/attacks/semantic.py)
| Saliency Map Method | https://arxiv.org/pdf/1511.07528.pdf | [Ongoing](https://github.com/iArunava/scratchai/blob/master/scratchai/attacks/attacks/saliency_map_method.py) |
| Fast Gradient Method | https://arxiv.org/abs/1412.6572 | [Implementation](https://github.com/iArunava/scratchai/blob/master/scratchai/attacks/attacks/fast_gradient_method.py) |
|Projected Gradient Descent | https://arxiv.org/pdf/1607.02533.pdf <br/> https://arxiv.org/pdf/1706.06083.pdf | [Implementation](https://github.com/iArunava/scratchai/blob/master/scratchai/attacks/attacks/fast_gradient_method.py) |
|DeepFool | https://arxiv.org/abs/1511.04599 [pdf](https://arxiv.org/pdf/1511.04599.pdf) | [Implementation](https://github.com/iArunava/scratchai/blob/master/scratchai/attacks/attacks/deepfool.py) |
  
  
## Tutorials

Tutorials on how to get the most out of scratchai can be found here: https://github.com/iArunava/scratchai/tree/master/tutorials

These are ongoing list of tutorials and scratchai is looking for more and more contributions. If you are willing to contribute 
please take a look at the `CONTRIBUTING.md` / open a issue.

## License
The code under this repository is distributed under MIT License. Feel free to use it in your own work with proper citations to this repository.
