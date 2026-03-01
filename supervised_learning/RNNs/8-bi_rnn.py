#!/usr/bin/env python3
"""This module includes the function bi_rnn that
performs forward propagation for a bidirectional RNN"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Args:
        bi_cell is an instance of BidirectinalCell
         that will be used for the forward propagation
        X is the data to be used, given as a
         numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
        h_0 is the initial hidden state in the forward direction,
         given as a numpy.ndarray of shape (m, h)
        h is the dimensionality of the hidden state
        h_t is the initial hidden state in the backward direction,
         given as a numpy.ndarray of shape (m, h)
    Returns:
        H, Y
        H is a numpy.ndarray containing all the concatenated hidden states
        Y is a numpy.ndarray containing all the outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    H_backward = []
    H_forward = []
    h_prev = h_0
    for step in range(t):
        # Get input at current time step
        x_t = X[step]
        h_next = bi_cell.forward(h_prev, x_t)
        H_forward.append(h_next)
        h_prev = h_next

    h_next = h_t
    # Backward pass: process from time t-1 to 0
    for step in range(t - 1, -1, -1):
        x_t = X[step]
        h_prev = bi_cell.backward(h_next, x_t)

        # Store the backward hidden state (at the beginning since
        # we're going backward)
        H_backward.insert(0, h_prev)
        h_next = h_prev

    H = []
    for step in range(t):
        h_concat = np.concatenate((H_forward[step],
                                   H_backward[step]), axis=1)
        H.append(h_concat)
    # Converting H to numpy array: shape (t, m, 2*h)
    H = np.array(H)
    Y = bi_cell.output(H)

    return H, Y
