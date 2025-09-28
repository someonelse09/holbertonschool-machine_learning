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
    kh, kw, kc = kernel.shape
    sh, sw = stride
    ph, pw = 0, 0

    if kc != c:
        raise ValueError("Kernel channel dimension must match image channels")

    # Determine padding amounts (we compute top/bottom and left/right separately
    # so that odd total padding is split correctly)
    if isinstance(padding, tuple) and len(padding) == 2:
        ph, pw = padding
        pad_top, pad_bottom = ph, ph
        pad_left, pad_right = pw, pw
    elif padding == 'same':
        # desired output sizes (ceil for stride > 1)
        out_h = int(np.ceil(h / sh))
        out_w = int(np.ceil(w / sw))

        # total padding needed along height/width
        pad_along_height = max((out_h - 1) * sh + kh - h, 0)
        pad_along_width = max((out_w - 1) * sw + kw - w, 0)

        # split into top/bottom and left/right (bottom/right gets the extra if odd)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        # keep ph/pw as "nominal" symmetric amounts (for compatibility)
        ph = pad_top
        pw = pad_left
    elif padding == 'valid':
        pad_top = pad_bottom = pad_left = pad_right = 0
        ph, pw = 0, 0
    else:
        raise ValueError("padding must be 'same', 'valid', or a tuple")

    # Pad the images (channels not padded)
    padded_images = np.pad(
        images,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant'
    )

    # Calculate output dimensions
    padded_h, padded_w = padded_images.shape[1], padded_images.shape[2]
    convolution_height = (padded_h - kh) // sh + 1
    convolution_width = (padded_w - kw) // sw + 1

    # Initialize output (single-channel result per image because channels are summed)
    convolved_image = np.zeros((m, convolution_height, convolution_width))

    for i in range(convolution_height):
        for j in range(convolution_width):
            # Extract patch from all images and channels
            patch = padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            # Convolve: sum over spatial dimensions (1,2) and channel dimension (3)
            convolved_image[:, i, j] = np.sum(patch * kernel, axis=(1, 2, 3))

    return convolved_image
