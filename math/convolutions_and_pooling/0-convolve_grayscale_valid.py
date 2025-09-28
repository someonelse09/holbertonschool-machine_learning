#!/usr/bin/env python3
"""This module includes the function
that performs a valid convolution on grayscale images"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Args:
        images is a numpy.ndarray with shape
         (m, h, w) containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        kernel is a numpy.ndarray with shape
         (kh, kw) containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
        only two for loops are allowed to use;
         any other loops of any kind are not allowed
    Returns:
        a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    convolved_height = h - kh + 1
    convolved_width = w - kw + 1
    convolved_image = np.zeros((m, convolved_height, convolved_width))

    for i in range(convolved_height):
        for j in range(convolved_width):
            patch = images[:, i:i + kh, j:j + kw]
            # size of the patch equals (m , kh, kw)
            # Performing elementwise multiplication and then adding the results
            convolved_image[:, i, j] += np.sum(patch * kernel, axis=(1, 2))
    return convolved_image
