#!/usr/bin/env python3
"""This module contains the function
that performs forward propagation for a simple RNN"""
import numpy as np
rnn_cell = __import__('0-rnn_cell').RNNCell


def rnn(rnn_cell, X, h_0):
    """
    Args:
        rnn_cell is an instance of RNNCell that will be used for the forward propagation
        X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
        h_0 is the initial hidden state, given as a numpy.ndarray of shape (m, h)
        h is the dimensionality of the hidden state
    Returns:
        H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    H = []
    Y = []
    H.append(h_0)
    h_prev = h_0
    for k in range(t):
        # Get input at current time step
        x_t = X[k]
        h_next, y_t = rnn_cell.forward(h_prev, x_t)
        H.append(h_next)
        Y.append(y_t)
        h_prev = h_next
    return np.array(H), np.array(Y)
