"""
The Projected Gradient Descent attack.
"""

import numpy as np
import torch
from scratchai.attacks import FGD
from scratchai.attacks.utils import clip_eta

class ProjectedGradientDescent(Attack):
    """
    This class implements either the Basic Iterative Method
    (Kuarkin et al. 2016) when rand_init is set to 0. or the
    Madry et al. (2017) method when rand_minmax is larger than 0.
    Paper link (Kuarkin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
    Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf

    Args:
        model: Model
        dtype: dtype of the data
        default_rand_init: whether to use random initialization by default
        kwargs: passed through to super constructor
    """

    FGM_CLASS = FGD

    def __init__(self, model, dtype='float32', default_rand_init=True, **kwargs):
        """
        Create a ProjectedGradientDescent instance.
        """

        super(ProjectedGradientDescent, self).__init__(model, dtype, **kwargs)

        self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target', 'clip_min',
                                'clip_max')
        
        self.structural_kwargs = ['ord', 'nb_iter', 'rand_init', 'sanity_checks']
        self.default_rand_init = default_rand_init


    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.

        Args:
            x: The model's symbolic inputs.
            kwargs: See `parse_params`
        """

        assert self.parse_params(**kwargs)

        asserts = []

        # If a data range was specified, check that the input was in that range
        if self.clip_min is not None:
            asserts.append(x.any() >= self.clip_min)

        if self.clip_max is not None:
            asserts.append(x.any() <= self.clip_max)

        # Initialize loop variables
        if self.rand_init:
            eta = torch.FloatTensor(*x.shape).uniform_(-self.minmax, self.minmax)
        else:
            eta = torch.zeros_like(x)

        # Clip eta
        eta = clip_eta(eta, self.ord, self.eps)
