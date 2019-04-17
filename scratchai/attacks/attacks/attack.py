"""
Base Class for PyTorch Attacks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Attack']

class Attack():
    """
    This class is the base class for all the PyTorch Attacks

    Args:
        model: The model nn.Module object
        dtype: A string mentioning the data type of the model
    """
    def __init__(self, model:nn.Module, dtype:str):
        self.model = model
        self.dtype = dtype

    def get_or_guess_labels(self, x, kwargs):
        """
        Get the label to use in generating an adversarial example for x.
        The kwargs are fed directly from the kwargs of the attack.

        If 'y' is in kwargs, then assume its an untargetted attack and use
        that as the label.

        If 'y_target' is in kwargs and is not None, then assume it's a 
        targetted attack and use that as the label.

        Otherwise, use the model's prediction as the label and perform an 
        untargetted attack.
        """

        if 'y' in kwargs and 'y_target' in kwargs:
            raise ValueError("Cannot set both 'y' and 'y_target'")
        elif 'y' in kwargs:
            labels = kwargs['y']
        elif 'y_target' in kwargs:
            labels = kwargs['y_target']
        else:
            # TODO Make sure softmax outputs are not again 
            # passed through softmax layer
            # TODO Make sure this function is implemented as expected
            logits = self.model(x if len(x.shape) == 4 else x.unsqueeze(0))
            pred_max = torch.argmax(logits, dim=1)
            opreds = torch.float(logits == pred_max)
            opreds.requires_grad = False
        
        nb_classes = opreds.size(1)
        return opreds, nb_classes
