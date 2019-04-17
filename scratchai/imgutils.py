import numpy as np
import matplotlib.pyplot as plt

def thresh_img(img:np.ndarray, rgb, tcol:list=[0, 0, 0]):
    """
    This function is used to threshold the image to a certain color.

    Args:
        img (np.ndarray): The image
        rgb (tuple or list): The color below which all colors will be blacked.
        tcol (list): The color to be filled in the thresholded part of
                     the image. Defaults to [0, 0, 0].

    Returns:
        np.ndarray: The threshold image
    """

    assert type(img) == np.ndarray
    assert type(rgb) == tuple or type(rgb) == list
    assert type(tcol) == list

    img = np.copy(img)
    tidx = (img[:, :, 0] < rgb[0]) \
           | (img[:, :, 1] < rgb[1]) \
           | (img[:, :, 2] < rgb[2])
    img[tidx] = tcol
    return img

def mask_reg(img, pnts, reln, deg=1, tcol=[0, 0, 0]):
    """
    Region Masking.

    Given a set of points which corrsponds to a polygon this
    function masks that polygon and returns the masked image.
    
    Arguments
    ---------
    img : np.ndarray, shape (H, W, 3)
          The image which needs to be masked.
    pnts : list
           A list containing the set of points (x, y)
           where each is a list / tuple.
    reln : list
           Where each element is a list/tuple (n1, n2)
           where n1 < len(points) and n2 < len(points) and it denotes
           that these two points are connected. 
           Do note: that n1 corresponds to the n1th element in points.
    deg : int
          Degree of the fitting polynomial.
    tcol : list
          A list containing the rgb values for the
          color which the mask needs to be filled with.
          Defaults to [0, 0, 0]

    Returns
    -------
    img : np.ndarray
          The masked image.
    """

    assert type(img) == np.ndarray
    for param in [pnts, reln, tcol]:
        assert type(param) == list or type(param) == tuple
    
    img = np.copy(img)
    h, w = img.shape
    
    # Fit the lines
    psolns = {}
    for ii, p1, p2 in enumerate(reln):
        soln = np.polyfit((p1[0], p2[0]), (p1[1], p2[1]), deg)
        psolns[ii] = soln
    
    # Find the region inside the lines
    xx, yy = np.meshgrid(np.arange(h), np.arange(w))
    reg_thresh = np.zeros_like(xx)
    for _, val in psolns.items():
        reg_thresh &= (yy > (xx*val[0] + val[1]))
    
    # Color pixels which are inside the region of interest
    img[reg_thresh] = tcol

    return img
