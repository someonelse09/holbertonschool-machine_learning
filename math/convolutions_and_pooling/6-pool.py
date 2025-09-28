#!/usr/bin/env python3
"""This module includes the function
that performs a convolution on grayscale images"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Args:
        images is a numpy.ndarray with shape
         (m, h, w, c) containing multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
        kernel_shape is a tuple of (kh, kw)
         containing the kernel shape for the pooling
        kh is the height of the kernel
        kw is the width of the kernel
        stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
        mode indicates the type of pooling
        max indicates max pooling
        avg indicates average pooling
        You are only allowed to use two for loops;
         any other loops of any kind are not allowed
    Returns:
        a numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    convolution_height = (h - kh) // 2 + 1
    convolution_width = (w - kw) // 2 + 1

    pooled_images = np.zeros((m, convolution_height, convolution_width, c))
    for i in range(convolution_height):
        for j in range(convolution_width):
            pool_region = images[:, i*sh:i*sh +kh, j*sw:j*sw + kw, :]

            if mode == 'max':
                pooled_images[:, i, j, :] = np.max(pool_region, axis=(1, 2))
            elif mode == 'avg':
                pooled_images[:, i, j, :] = np.mean(pool_region, axis=(1, 2))
            else:
                raise ValueError("mode must be 'max' or 'avg'")
    return pooled_images
