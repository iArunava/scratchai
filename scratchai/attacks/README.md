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

| Attack | Alexnet | VGG16 | VGG19 | Resnet18 |
| :----- | :---: | :---: | :---: | :---: |
| Noise  | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> </table> |
| Semantic  | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> </table> |
| Fast Gradient Sign Method | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> </table> |
| Projected Gradient Descent  | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> </table> | <table> <tr> <th> Dataset </th> <th> Acc@1 w/ </th> <th> Acc@5 w/ </th> <th> Acc@1 w/o </th> <th> Acc@5 w/o </th> </tr> </table> |
