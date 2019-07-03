"""
Utility Functions as required by the models.
"""
import torch

from scratchai.utils import load_pretrained


__all__ = ['get_net', 'transfer_weights', 'transfer_weights_with_sd']


def get_net(net, pretrained=True, pretrain_url=None, fname=None, 
            kwargs_net=None, **kwargs_load_pretrained):
  """
  Function to load a model and do some required cutting and splitting on it.

  Arguments
  ---------
  net                    : scratchai.nets.*
                           The class which to initialize, 
                           *not an initialized net.*

  pretrained             : bool
                           Whether to load a pretrained net or not.

  pretrain_url           : str
                           The url from which to load a pretrained model.
                           It should be just the file id, if the file is 
                           hosted on Google Drive.

  fname                  : str
                           The file name with which to store the pretrained 
                           file.

  kwargs_net             : dict
                           The extra parameters which are passed while 
                           initializing the net

  kwargs_load_pretrained : dict
                           The extra parameters passed to load_pretrained.
  """
  cust_nc = kwargs_net['nc'] if 'nc' in kwargs_net else None
  if pretrained and 'nc' in kwargs_net: kwargs_net.pop('nc')
  net = net(**kwargs_net)
  if pretrained:
    return load_pretrained(net, pretrain_url, fname, nc=cust_nc, 
                           **kwargs_load_pretrained)
  return net


# No tests written
def transfer_weights(net1, net2):
  """
  Transfer weights from net1 to net2,
  Even when the keys doesn't match!!!

  Arguments
  ---------
  net1 : nn.Module
         The net whoose weights needs to be transfered.

  net2 : nn.Module
         The net to which the weights needs to be transfered.

  Notes
  -----
  Even though the keys doesn't need to match.
  The shape for each weight tensor should match!
  """
  sd1 = net.state_dict()
  sd2 = net.state_dict()
  
  sd2 = transfer_weights_with_sd(sd1, sd2)
  net2.load_state_dict(sd2)

  # TODO Check if the return statement is needed.
  return net2


# No tests written
def transfer_weights_with_sd(sd1, sd2):
  """
  This function allows to transfer the weights using just
  the state dicts. And no usage of the model as a whole.

  Arguments
  ---------
  sd1 : OrderedDict
        The state_dict for the first net (from which to transfer)

  sd2 : OrderedDict
        The state dict for the second net (to which to transfer)
  """

  new_dict = {}
  for key1, key2 in zip(sd1.keys(), sd2.keys()):
    new_dict[key2] = sd1[key1].data.clone()

  sd2.update(new_dict)
  return sd2
