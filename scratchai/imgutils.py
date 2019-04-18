import numpy as np
import matplotlib.pyplot as plt
import cv2
import operator

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

def mask_reg(img, pnts, reln, deg:int=1, tcol:tuple=(0, 0, 0), 
             locate:bool=False, invert:bool=False) -> np.ndarray:
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
    tcol : tuple of 3 elements
          A list containing the rgb values for the
          color which the mask needs to be filled with.
          Defaults to (0, 0, 0)
    locate : bool
             Denotes if to mark the points also in the image. 
             Defaults to False.
    invert : bool
             Denotes whether to invert the final indexes. Defaults to False.

    Returns
    -------
    img : np.ndarray
          The masked image.
    """

    assert type(img) == np.ndarray
    for param in [pnts, reln, tcol]:
        assert type(param) == list or type(param) == tuple
    
    img = np.copy(img)
    h, w = img.shape[1], img.shape[0]
    ops = {'<' : operator.lt, '>' : operator.gt}
    
    # Fit the lines
    psolns = {}
    for ii, (p1, p2, rel) in enumerate(reln):
        pnt1 = pnts[p1]; pnt2 = pnts[p2]
        soln = list(np.polyfit((pnt1[0], pnt2[0]), (pnt1[1], pnt2[1]), deg))
        soln.append(rel)
        psolns[ii] = soln

    # Find the region inside the lines
    xx, yy = np.meshgrid(np.arange(h), np.arange(w))
    for ii, val in psolns.items():
        # TODO Give option to choose btw bitwise_and and bitwise_or
        reg_thresh = ops[val[2]](yy, (xx*val[0] + val[1])) if not ii else \
                     reg_thresh & ops[val[2]](yy, (xx*val[0] + val[1]))
            
    # Color pixels which are inside the region of interest
    img[reg_thresh == False if invert else reg_thresh] = tcol
    
    return img if not locate else mark_pnt_on_img(img, pnts)

def mark_pnt_on_img(img, pnts:list, col:tuple=(0, 255, 0)) -> np.ndarray:
    """
    Mark points on Image.

    Arguments
    ---------
    img : np.ndarray - Shape like (H x W x 3)
          The image on which the points need to be marked.
    pnts : list of tuples
           where each tuple is (x, y) a point to mark.
    col : tuple of 3 elements
          which are the rgb values for the color of the point.
    Returns
    -------
    img : np.ndarray
          With all the points marked on the image
    """

    img = np.copy(img)
    for pnt in pnts:
        cv2.circle(img, pnt, 10, col, -1)
    return img
