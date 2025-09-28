#!/usr/bin/env python3
"""This module includes the function
that performs a convolution on grayscale images"""

import numpy as np


def convolve(images, kernel, padding='same', stride=(1, 1)):
    """
    Args:
        images is a numpy.ndarray with shape
         (m, h, w, c) containing multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
        kernels is a numpy.ndarray with shape
         (kh, kw, c, nc) containing the kernels for the convolution
        kh is the height of a kernel
        kw is the width of a kernel
        nc is the number of kernels
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
        only three for loops are allowed to use;
         any other loops of any kind are not allowed
    Returns:
        a numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernel.shape
    sh, sw = stride
    ph, pw = 0, 0

    if kc != c:
        raise ValueError("Kernel channel dimension must match image channels")

    if isinstance(padding, tuple) and len(padding) == 2:
        ph, pw = padding
    elif padding == 'same':
        ph = int(np.ceil(((h - 1)*sh + kh - h) / 2))  # or just use //
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
                           (pw, pw),
                           (0, 0)),
                          mode='constant')

    convolved_image = np.zeros((m, convolution_height, convolution_width, nc))

    for i in range(convolution_height):
        for j in range(convolution_width):
            for k in range(nc):
                patch = padded_image[:, i*sh:i*sh + kh, j*sw:j*sw + kw, :]
                current_kernel = kernel[:, :, :, k]
                convolved_image[:, i, j, k] += np.sum(
                    patch * current_kernel, axis=(1, 2, 3)
                )
    return convolved_image
