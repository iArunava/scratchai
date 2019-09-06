# Generative Adversarial Networks (GANs)

This file lists some tips and tricks and/or some info about GANs

## A few things observed while implementing DCGAN

1. The D gets powerful much faster than the G. (when training the D for every epoch in G, this even happends after random initializing the D after 25 epochs.)

2. After 25 epochs, if the D is randomly initialized, the G seems to get more powerful, much faster. (Needs to be checked once again)
