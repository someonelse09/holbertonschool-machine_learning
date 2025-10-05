#!/usr/bin/env python3
"""
This module includes the function that performs back propagation
over a pooling layer of a neural network
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network

    Args:
        dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
        the partial derivatives with respect to the output of the pooling layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c_new is the number of channels
        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing
        the output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c is the number of channels
        kernel_shape: tuple of (kh, kw) containing the size of the kernel
        for the pooling
        kh is the kernel height
        kw is the kernel width
        stride is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
        mode is a string containing either 'max' or 'avg', indicating whether
        to perform maximum or average pooling

    Returns:
        dA_prev: the partial derivatives with respect to the previous layer
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize dA_prev with zeros
    dA_prev = np.zeros_like(A_prev)

    for example in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c_new):
                    if mode == 'max':
                        # Extract the current window from A_prev
                        window = A_prev[example, i*sh:i*sh+kh, j*sw:j*sw+kw, k]

                        # Create a mask from the maximum value
                        mask = (window == np.max(window))

                        # Distribute the gradient to the position(s) of the max value
                        dA_prev[example, i*sh:i*sh+kh, j*sw:j*sw+kw, k] += \
                            mask * dA[example, i, j, k]

                    elif mode == 'avg':
                        # Calculate the average gradient value
                        avg_gradient = dA[example, i, j, k] / (kh * kw)

                        # Distribute the gradient equally across the window
                        dA_prev[example, i*sh:i*sh+kh, j*sw:j*sw+kw, k] += \
                            np.ones((kh, kw)) * avg_gradient

                    else:
                        raise ValueError("mode must be either 'max' or 'avg'")

    return dA_prev
