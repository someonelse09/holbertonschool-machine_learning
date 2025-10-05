#!/usr/bin/env python3
"""This module includes the function that
performs back propagation over a
convolutional layer of a neural network"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Args:
        dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new)
         containing the partial derivatives with respect to
          the unactivated output of the convolutional layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c_new is the number of channels in the output
        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
         containing the output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
        W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
         containing the kernels for the convolution
        kh is the filter height
        kw is the filter width
        b is a numpy.ndarray of shape (1, 1, 1, c_new)
         containing the biases applied to the convolution
        padding is a string that is either same or valid,
         indicating the type of padding used
        stride is a tuple of (sh, sw)
         containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
        may import numpy as np
    Returns:
        the partial derivatives with respect to the
         previous layer (dA_prev), the kernels (dW),
          and the biases (db), respectively
    """
    m, h_new, w_new, c_new = dZ.shape
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
        raise ValueError("padding can be either same or valid")
    A_padded = np.pad(A_prev, ((0, 0),
                               (ph, ph),
                               (pw, pw),
                               (0, 0)),
                      mode='constant')
    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    dA_padded = np.zeros((m, h_prev + 2*ph, w_prev + 2*pw, c_prev))
    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                patch = A_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                sl = dZ[:, i, j, k][:, np.newaxis, np.newaxis, np.newaxis]
                dW[:, :, :, k] += np.sum(patch * sl, axis=0)
                dA_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :] += \
                    W[:, :, :, k] * sl
    # removing padding from dA_padded
    if padding == 'same':
        dA_prev = dA_padded[:, ph:ph+h_prev, pw:pw+w_prev, :]
    else:
        dA_prev = dA_padded
    return dA_prev, dW, db


