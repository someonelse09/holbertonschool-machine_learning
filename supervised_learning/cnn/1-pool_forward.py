#!/usr/bin/env python3
"""This module includes the function
that performs forward propagation over
a pooling layer of a neural network"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Args:
        A_prev is a numpy.ndarray of shape
         (m, h_prev, w_prev, c_prev) containing
          the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
        kernel_shape is a tuple of (kh, kw)
         containing the size of the kernel for the pooling
        kh is the kernel height
        kw is the kernel width
        stride is a tuple of (sh, sw)
         containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
        mode is a string containing either
         max or avg, indicating whether to perform
          maximum or average pooling, respectively
        you may import numpy as np
    Returns:
        the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    pooling_height = ((h_prev - kh) // sh) + 1
    pooling_width = ((w_prev - kw) // sw) + 1
    pooled_layer = np.zeros((m, pooling_height, pooling_width, c_prev))

    for i in range(pooling_height):
        for j in range(pooling_width):
            pool_region = A_prev[:, i*sh:i*sh+kh, j*sw:j*sw + kw, :]
            if mode == 'max':
                pooled_layer[:, i, j, :] = np.max(pool_region,
                                                  axis=(1, 2))
            elif mode == 'avg':
                pooled_layer[:, i, j, :] = np.mean(pool_region,
                                                   axis=(1, 2))
            else:
                raise ValueError("mode can be either max or avg")
    return pooled_layer
