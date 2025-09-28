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
    
    if isinstance(padding, tuple) and len(padding) == 2:
        ph, pw = padding
    elif padding == 'same':
        # Calculate padding for 'same' convolution
        # For stride=1, output size should equal input size
        # For stride>1, it's more complex, but basic formula:
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
        
        # Handle asymmetric padding for even kernels (extra at end, not beginning)
        ph_extra = (kh - 1) % 2
        pw_extra = (kw - 1) % 2
        
        # Standard convention: put extra padding at the end (bottom-right)
        padded_images = np.pad(images, ((0, 0), (ph, ph + ph_extra), (pw, pw + pw_extra), (0, 0)), mode='constant')
    elif padding == 'valid':
        ph, pw = 0, 0
        padded_images = images  # No padding for valid
    else:
        raise ValueError("padding must be 'same', 'valid', or a tuple")
    
    # For custom padding (tuple case)
    if isinstance(padding, tuple):
        padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    
    # Calculate output dimensions
    padded_h, padded_w = padded_images.shape[1], padded_images.shape[2]
    convolution_height = (padded_h - kh) // sh + 1
    convolution_width = (padded_w - kw) // sw + 1
    
    # Initialize output (note: single channel output since we sum across all input channels)
    convolved_image = np.zeros((m, convolution_height, convolution_width))
    
    for i in range(convolution_height):
        for j in range(convolution_width):
            # Extract patch from all images and channels
            patch = padded_images[:, i*sh:i*sh + kh, j*sw:j*sw + kw, :]
            # Convolve: sum over spatial dimensions (1,2) and channel dimension (3)
            convolved_image[:, i, j] = np.sum(patch * kernel, axis=(1, 2, 3))
    
    return convolved_image
