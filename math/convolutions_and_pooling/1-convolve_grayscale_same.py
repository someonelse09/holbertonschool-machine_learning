#!/usr/bin/env python3
"""This module includes the function that
performs a same convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
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
        if necessary, the image should be padded with 0’s
        only two for loops are allowed to use;
         any other loops of any kind are not allowed
    Returns:
        a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = (kh - 1) // 2
    pw = (kw - 1) // 2

    # For even kernel we may need to add one additional pixel
    ph_extra = (kh - 1) % 2
    pw_extra = (kw - 1) % 2

    # Padded image should give result image
    #  having the same dimensions as original one
    # w + 2*p - kw + 1 = w --> p = (kw - 1) / 2

    padded_image = np.pad(images, ((0, 0),
                                   (ph + ph_extra, ph),
                                   (pw + pw_extra, pw)),
                          mode='constant')
    convolved_image = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            patch = padded_image[:, i:i + kh, j:j + kw]
            convolved_image[:, i, j] += np.sum(patch * kernel, axis=(1, 2))
    return convolved_image
