#!/usr/bin/env python3
"""This module includes the function
that performs a convolution on grayscale images"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
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
        padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
        ph is the padding for the height of the image
        pw is the padding for the width of the image
        the image should be padded with 0’s
        stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
        only two for loops are allowed to use;
         any other loops of any kind are not allowed
    Returns:
        a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    ph, pw = 0, 0

    if isinstance(padding, tuple) and len(padding) == 2:
        ph, pw = padding
    elif padding == 'same':
        ph = int(np.ceil(((h - 1)*sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1)*sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        raise ValueError("padding must be 'same', 'valid', or a tuple")

    convolution_height = int(((h + 2*ph - kh) / sh) + 1)
    convolution_width = int(((w + 2*pw - kw) / sw) + 1)
    padded_image = np.pad(images,
                          ((0, 0),
                           (ph, ph),
                           (pw, pw)),
                          mode='constant')
    convolved_image = np.zeros((m, convolution_height, convolution_width))

    for i in range(convolution_height):
        for j in range(convolution_width):
            patch = padded_image[:, i*sh:i*sh + kh, j*sw:j*sw + kw]
            convolved_image[:, i, j] += np.sum(patch * kernel, axis=(1, 2))
    return convolved_image

