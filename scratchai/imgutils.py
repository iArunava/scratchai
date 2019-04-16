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
