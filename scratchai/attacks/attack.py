"""
The Attack Interface
"""

import numpy as np
import torch
import torch.nn as nn

class Attack():
    """
    Abstract base class for all attack classes
    """

    def __init__(self, model, dtype='float', **kwargs):
        """
        Args:
            model: An instance of the cleverhans model class
            dtype: The floating point precision to use
        """

        self.dtype = dtype
        self.np_dtype = np(dtype)
        self.model = model

        # To keep a track of old graphs and cache them
        self.graphs = {}

        # When calling generate_np, arguments in the following set should be
        # fed into the graph, as they are not structural items that require 
        # generating a new graph
        # This dict should map names of arguments to the types they should have
        self.feedable_kwargs = tuple()

        # When calling generate_np, arguments in the following set should NOT
        # be fed into the graph, as they are not structural items that require
        # generating a new graph
        # This list should contain the names of the structural arguments.
        self.structural_kwargs = []

    def generate(self, x, **kwargs):
        """
        Generate the attack's symbolic graph for adversarial examples. This method should
        be overridden in any child class that implements an attack that is expressible symbolically.
        Otherwise, it will wrap the numerical implementation as a symbolic operator.

        Args:
            x: The model's symbolic inputs
            **kwargs: optional parameters used by child classes.
                Each child class defines additional parameters as needed.
                Child classes that use the following concepts should use the 
                following names:
                    
                    clip_min: minimum feature value
                    clip_max: maximum feature value
                    eps: size of norm constraint on adversarial perturbation
                    ord: order of norm constraint
                    nb_iter: number of iterations
                    eps_iter: size of norm constraint on iteration
                    y_target: if specified, the attack is targetted
                    y: Do not specify if y_target is specified.
                        If specified, the attack is untargeted, aims to make the output
                        class not to be y.
                        If neither y_target nor y is specified, y is inferred by having the
                        model classify the input.
                For other concepts, its generally a good idea to read other classes and check
                for name consistency.
        :return: A symbolic representation of the adversarial examples.
        """

        error = 'Sub-classes  must implement generate'
        raise NotImplementedError(error)
        return x

    def generate_np(self, x_val, **kwargs):
        """
        Generate adversarial examples and return them as NumPy array.
        Sub-classes *should not* implement this method unless they must
        perform special handling of arguments.

        Args:
            x_val: A Numpy array with the optional inputs
            **kwargs: optional parameters used by child classes.
        :return: A NumPy array holding the adversarial examples
        """
        # TODO
        return x_val
