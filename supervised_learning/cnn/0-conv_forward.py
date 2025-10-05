#!/usr/bin/env python3
"""This module includes the function that
performs forward propagation over
a convolutional layer of a neural network"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Args:
        A_prev is a numpy.ndarray of shape
         (m, h_prev, w_prev, c_prev) containing
          the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
        W is a numpy.ndarray of shape
         (kh, kw, c_prev, c_new) containing
          the kernels for the convolution
        kh is the filter height
        kw is the filter width
        c_prev is the number of channels in the previous layer
        c_new is the number of channels in the output
        b is a numpy.ndarray of shape
         (1, 1, 1, c_new) containing the
          biases applied to the convolution
        activation is an activation function applied to the convolution
        padding is a string that is either same
         or valid, indicating the type of padding used
        stride is a tuple of (sh, sw)
         containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
        you may import numpy as np
    Returns:
        the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    ph, pw = 0, 0
    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1)*sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1)*sw + kw - w_prev) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        raise ValueError("padding should be either same or valid")
    padded_image = np.pad(A_prev,
                         ((0, 0),
                         (ph, ph),
                         (pw, pw),
                         (0, 0)),
                         mode='constant')
    convolution_height = int((h_prev + 2 * ph - kh) / sh) + 1
    convolution_width = int((w_prev + 2 * pw - kw) / sw) + 1
    convolved_layer = \
        np.zeros((m, convolution_height, convolution_width, c_new))
    for i in range(convolution_height):
        for j in range(convolution_width):
            for k in range(c_new):
                patch = padded_image[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                convolved_layer[:, i, j, k] = np.sum(patch * W[:, :, :, k],
                                                     axis=(1, 2, 3))
    convolved_layer = activation(convolved_layer + b)
    return convolved_layer
