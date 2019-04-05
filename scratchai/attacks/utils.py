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
        eta = clip_by_value(eta, -eps, eps)
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
