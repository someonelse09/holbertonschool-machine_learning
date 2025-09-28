#!/usr/bin/env python3
"""This module includes the function that performs a convolution on images with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Args:
        images is a numpy.ndarray with shape (m, h, w, c) containing multiple images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        kernel is a numpy.ndarray with shape (kh, kw, c) containing the kernel for the convolution
            kh is the height of the kernel
            kw is the width of the kernel
        padding is either a tuple of (ph, pw), 'same', or 'valid'
            if 'same', performs a same convolution
            if 'valid', performs a valid convolution
            if a tuple:
                ph is the padding for the height of the image
                pw is the padding for the width of the image
            the image should be padded with 0's
        stride is a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
        only two for loops are allowed to use; any other loops of any kind are not allowed
    Returns:
        a numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, c = kernel.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))

    if padding == 'valid':
        ph = 0
        pw = 0

    if isinstance(padding, tuple):
        ph, pw = padding

    new_w = (w + 2 * pw - kw) // sw + 1
    new_h = (h + 2 * ph - kh) // sh + 1
    convolved = np.zeros((m, new_h, new_w))
    padded_images = np.pad(images,
                           pad_width=((0, 0), (ph, ph),
                                      (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)

    for i in range(new_h):
        for j in range(new_w):
            convolved[:, i, j] = np.sum(padded_images[:, i * sh:i * sh + kh,
                                        j * sw:j * sw + kw, :] *
                                        kernel, axis=(1, 2, 3))
    return convolved
