## Adversarial Machine Learning

Note: This section takes reference from the tf cleverhans library

Adversarial Machine Learning is the study of attacks and defenses that can be used to easily fool Machine Learning models and defend Machine Models against such attacks.

## Attacks

The attacks implemented in this section can be found here: https://github.com/iArunava/scratchai/README.md

## Benchmarks

This section performs benchmarking of all the attacks and defences implemented here.

The benchmarks reproduced here uses the ILSVC2012 Imagenet test set. The columns with Acc@n indicate the top-n accuracy 
and columns with `w/o` indicates the accuracy without the attack and the ones with `w` indicate the accuracy with the attack.

`NA` Indicates that it has not been measured yet.

| Attack | Lenet | Alexnet | VGG16 | VGG19 | Resnet18 |
| :----- | :---: | :---: | :---: | :---: | :---: |
| Noise | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> 0.984 </th> <th> 1.0 </th> <th> 0.9858 </th> <th> 1.0 </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> 0.9907 </th> <th> 1.0 </th> <th> 0.9908 </th> <th> 1.0 </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table>| <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table>| <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table> |
| Semantic  | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> 0.233 </th> <th> 0.645 </th> <th> 0.986 </th> <th> 1.0 </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> 0.278 </th> <th> 0.612 </th> <th> 0.99 </th> <th> 1.0 </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table>| <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table>| <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table> |
| Fast Gradient Sign Method  | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> 0.509 </th> <th> 0.993 </th> <th> 0.986 </th> <th> 1.0 </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> 0.831 </th> <th> 0.99 </th> <th> 0.99 </th> <th> 1.0 </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table>| <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table>| <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table> |
| Projected Gradient Descent  | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> 0.187 </th> <th> 0.982 </th> <th> 0.986 </th> <th> 1.0 </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> 0.667 </th> <th> 0.9984 </th> <th> 0.99 </th> <th> 1.0 </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table>| <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table>| <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> <tr> <th> MNIST </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> <tr> <th> ILSVRC2012 </th> <th> NA </th> <th> NA </th> <th> NA </th> <th> NA </th> </tr> </tr> </table> |

---

All the above benchmarks are done using the following code:

```
>>> from scratchai import *
>>> net = nets.lenet_mnist() # Get the network of choice (pretrained on the dataset)
>>> attacks.benchmark_atk(attacks.PGD, net, dset='mnist', bs=16, topk=(1, 5))

[INFO] Setting bs to 16.
[INFO] Setting trf to Compose(
          Resize(size=(256, 256), interpolation=PIL.Image.BILINEAR)
          CenterCrop(size=(224, 224))
          ToTensor()
          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          ).
[INFO] Setting dset to mnist.
[INFO] Setting root to ./.
[INFO] Setting topk to (1, 5).
[INFO] Setting dfunc to <class 'torchvision.datasets.folder.ImageFolder'>.
[INFO] Setting download to True.
[INFO] Net Frozen!
100%|_______________________________________________________| 625/625 [00:18<00:00, 34.71it/s]

Attack Summary on lenet with pgd attack:
---------------------------------------------
Top 1 original accuracy is 0.9858
Top 5 original accuracy is 1.0

-----------------------------------
Top 1 adversarial accuracy is 0.1874
Top 5 adversarial accuracy is 0.9818

```
