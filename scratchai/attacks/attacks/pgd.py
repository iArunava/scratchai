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
        adv_x = x + eta

        if self.clip_min is not None or self.clip_max is not None:
            adv_x = torch.clamp(adv_x, self.clip_max, self.clip_min)

        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            model_preds = self.model.get_probs(x)
            preds_max = reduce_max(model_preds, 1)
            y = torch.equals(model_preds, preds_max).float()
            y.requires_grad = False
            targeted = False
            del model_preds

        y_kwarg = 'y_target' if targeted else 'y'
        fgm_params = {
            'eps' : self.eps_iter,
            y_kwarg: y,
            'ord' : self.ord,
            'clip_min' : self.clip_min,
            'clip_max' : self.clip_max
        }

        if self.ord == 1:
            raise NotImplementedError("It's not clear that FGM is a good inner loop"
                                      " step for PGD when ord=1, because ord=1 FGM "
                                      " changes only one pixel at a time. We need "
                                      " to rigoursly test a strong ord=1 PGD "
                                      " before enabling this feature.")

        FGM = self.FGM_CLASS(self.model, dtype)

        while i < self.nb_iter:
            """
            Do a projected gradient step.
            """
            adv_x = FGM.generate(adv_x, **fgm_params)

            # Clipping perturbation eta to self.ord norm ball
            eta = adv_x - x
            eta = clip_eta(eta, self.ord, self.eps)
            adv_x = x + eta

            # Redo the clipping.
            # FGM alread already did it, but subtracting and re-adding eta can add some
            # small numerical error
            if self.clip_min is not None or self.clip_max is not None:
                adv_x = torch.clamp(adv_x, self.clip_min, self.clip_max)

        # Asserts run only on CPU
        # When multi-GPU eval code tries to force all PGD ops onto GPU, this
        # can cause an error.
        common_dtype = torch.float32
        # NOTE Maybe this needs a cast
        asserts.append(self.eps <= (1e6 + self.clip_max - self.clip_min))
        
        return adv_x
