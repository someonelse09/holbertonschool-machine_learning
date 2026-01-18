#!/usr/bin/env python3
"""This module contains the function that
performs forward propagation for a deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN

    Args:
        rnn_cells: a list of RNNCell
         instances of length l that will be used
                   for the forward propagation
            l is the number of layers
        X: the data to be used, given as
         a numpy.ndarray of shape (t, m, i)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0: the initial hidden state,
         given as a numpy.ndarray of shape (l, m, h)
            h is the dimensionality of the hidden state

    Returns:
        H: a numpy.ndarray containing all the hidden states
        Y: a numpy.ndarray containing all the outputs
    """
    t, m, i = X.shape
    length = len(rnn_cells)
    h = h_0.shape[2]

    # Initialize H to store all hidden
    # states for all layers across all time steps
    # Shape: (t+1, l, m, h) - includes initial states
    H = np.zeros((t + 1, length, m, h))

    H[0] = h_0

    # Initialize Y to store outputs
    # from the last layer at each time step
    # Shape: (t, m, o) where o is
    # determined by the last cell's output
    Y = []

    # Process each time step
    for step in range(t):
        # For the first layer, input is X[step]
        layer_input = X[step]

        # Process through each layer
        for layer in range(length):
            # Get previous hidden state for this layer
            h_prev = H[step, layer]

            # Forward propagation through current layer
            h_next, y = \
                rnn_cells[layer].forward(h_prev, layer_input)

            # Store the hidden state for
            # this layer at next time step
            H[step + 1, layer] = h_next

            # The output of this layer becomes
            # the input to the next layer
            layer_input = h_next

        # Store the output from the last layer
        # (which is y from the last iteration)
        Y.append(y)

    # Convert Y to numpy array: shape (t, m, o)
    Y = np.array(Y)

    return H, Y
