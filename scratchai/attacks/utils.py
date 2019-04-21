import torch
import numpy as np


def optimize_linear(grads, eps, ordr):
  """
  Solves for optimal input to a linear function under a norm constraint.

  Arguments
  ---------
  grads : torch.Tensor
           The gradients of the input.
  eps : float
        Scalar specifying the constraint region.
  ordr : [np.inf, 1, 2] 
         Order of norm constraint.

  Returns
  -------
  opt_pert : torch.Tensor
             Optimal Perturbation.
  """

  red_ind = list(range(1, len(grads.size())))
  azdiv = torch.tensor(1e-12, dtype=grads.dtype, device=grads.device)

  if ordr == np.inf:
    opt_pert = torch.abs(grads)

  elif ordr == 1:
    abs_grad = torch.abs(grads)
    sign = torch.sign(grads)
    ori_shape = [1] * len(grads.size())
    ori_shape[0] = grads.size(0)

    max_abs_grad, _ = torch.max(abs_grad.view(grads.size(0), -1), 1)
    max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).float()
    num_ties = max_mask
    for red_scalar in red_ind:
      num_ties = torch.sum(num_ties, red_scalar, keepdims=True)
    opt_pert = sign * max_mask / num_ties

  elif ordr == 2:
    # TODO
    pass

  scaled_pert = eps * opt_pert
  return scaled_pert


def clip_eta(eta, ord, eps):
    """
    Helper fucntion to clip the perturbation to epsilon norm ball.

    Args:
        eta: A tensor with the current perturbation
        ord: Order of the norm (mimics Numpy)
             Possible values: np.inf, 1 or 2.
        eps: Epsilon, bound of the perturbation.
    """

    # Clipping perturbation eta to self.ord norm ball
    if ord not in [np.inf, 1, 2]:
        raise ValueError('ord must be np.inf, 1, or 2.')

    reduce_ind = list(xrange(1, len(eta.shape)))
    azdiv = 1e-12

    if ord == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    else:
        if ord == 1:
            raise NotImplementedError("The expression below is not the correct way"
                                      " to project onto the L1 norm ball.")
            norm = torch.maximum(azdiv, torch.mean(torch.abs(eta), reduce_ind))

        elif ord == 2:
            # azdiv(avoid_zero_div) must go inside sqrt to avoid a divide by zero
            # in the gradient through this operation.
            norm = torch.sqrt(torch.maximum(azdiv, torch.mean(eta**2, reduce_ind)))
        
        # We must clip to within the norm ball, not 'normalize' onto the
        # surface of the ball
        factor = torch.minimum(1., div(eps, norm))
        eta *= factor

    return eta
