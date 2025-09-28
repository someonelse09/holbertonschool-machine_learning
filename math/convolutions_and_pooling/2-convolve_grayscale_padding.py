#!/usr/bin/env python3
"""This module includes the function that performs a
convolution on grayscale images with custom padding"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
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
        padding is a tuple of (ph, pw)
        ph is the padding for the height of the image
        pw is the padding for the width of the image
        the image should be padded with 0’s
        only two for loops are allowed to use;
         any other loops of any kind are not allowed
    Returns:
        a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    fixed_h = h + 2 * ph - kh + 1
    fixed_w = w + 2 * pw - kw + 1

    padded_image = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    convolved_image = np.zeros((m, fixed_h, fixed_w))
    for i in range(fixed_h):
        for j in range(fixed_w):
            patch = padded_image[:, i:i + kh, j:j + kw]
            convolved_image[:, i, j] += np.sum(patch * kernel, axis=(1, 2))
    return convolved_image
