## Adversarial Machine Learning

Note: This section takes reference from the tf cleverhans library

Adversarial Machine Learning is the study of attacks and defenses that can be used to easily fool Machine Learning models and defend Machine Models against such attacks.

## Attacks

The attacks implemented in this section can be found here: https://github.com/iArunava/scratchai/README.md

## Benchmarks

This section performs benchmarking of all the attacks and defences implemented here.

The benchmarks reproduced here uses the ILSVC2012 Imagenet test set. The columns with Acc@n indicate the top-n accuracy 
and columns with `w/o` indicates the accuracy without the attack and the ones with `w` indicate the accuracy with the attack.

The table lists information in the following way:

| Attack | Model |
| :----- | :-----: |
| Attack Name | top-1 acc of the model w/o attack <br/> top-5 acc of the model w/o attack <br/> top-1 acc of the model w/ attack <br/> top-5/ acc of the model w/ attack |

| Attack | Alexnet | VGG16 | VGG19 | Resnet18 |
| :----- | :---: | :---: | :---: | :---: |
| Noise  | acc@1 w/o<br/>acc@5 w/o<br/>acc@1 w<br/>acc@5 w | acc@1 w/o<br/>acc@5 w/o<br/>acc@1 w<br/>acc@5 w | acc@1 w/o<br/>acc@5 w/o<br/>acc@1 w<br/>acc@5 w | acc@1 w/o<br/>acc@5 w/o<br/>acc@1 w<br/>acc@5 w |
| Semantic | acc@1 w/o<br/>acc@5 w/o<br/>acc@1 w<br/>acc@5 w | acc@1 w/o<br/>acc@5 w/o<br/>acc@1 w<br/>acc@5 w | acc@1 w/o<br/>acc@5 w/o<br/>acc@1 w<br/>acc@5 w | acc@1 w/o<br/>acc@5 w/o<br/>acc@1 w<br/>acc@5 w |
| Fast Gradient Sign Method  | acc@1 w/o<br/>acc@5 w/o<br/>acc@1 w<br/>acc@5 w | acc@1 w/o<br/>acc@5 w/o<br/>acc@1 w<br/>acc@5 w | acc@1 w/o<br/>acc@5 w/o<br/>acc@1 w<br/>acc@5 w | acc@1 w/o<br/>acc@5 w/o<br/>acc@1 w<br/>acc@5 w |
| Projected Gradient Descent  | acc@1 w/o<br/>acc@5 w/o<br/>acc@1 w<br/>acc@5 w | acc@1 w/o<br/>acc@5 w/o<br/>acc@1 w<br/>acc@5 w | acc@1 w/o<br/>acc@5 w/o<br/>acc@1 w<br/>acc@5 w | acc@1 w/o<br/>acc@5 w/o<br/>acc@1 w<br/>acc@5 w |
