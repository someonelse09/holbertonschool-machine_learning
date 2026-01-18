#!/usr/bin/env python3
"""This module contains the function
that performs forward propagation for a simple RNN"""
import numpy as np


class RNNCell:
    """A single cell of Recurrent Neural Network"""
    def __init__(self, i, h, o):
        """Constructor
        Args:
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs
            Creates the public instance attributes Wh, Wy, bh,
             by that represent the weights and biases of the cell
            Wh and bh are for the concatenated hidden state and input data
            Wy and by are for the output
            The weights should be initialized using a random
            normal distribution in the order listed above
            The weights will be used on right side for matrix multiplication
            The biases should be initialized as zeros"""
        # Wh: weight matrix for concatenated [h_prev, x_t] -> h_next
        # Shape: (i + h, h) because we concat input(i)
        # and hidden(h), output is hidden(h)
        self.Wh = np.random.randn(i+h, h)

        # Wy: weight matrix for hidden state -> output
        # Shape: (h, o) because input is hidden(h), output is output(o)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Args:
            x_t is a numpy.ndarray of shape (m, i)
            that contains the data input for the cell
            m is the batche size for the data
            h_prev is a numpy.ndarray of shape (m, h)
            containing the previous hidden state
            The output of the cell should use a softmax activation function
        Returns:
             h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        # Concatenate h_prev and x_t along axis 1 (features)
        # h_prev shape: (m, h), x_t shape:
        # (m, i) -> concat shape: (m, h+i)
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute next hidden state with tanh activation
        # concat @ Wh: (m, h+i) @ (h+i, h) -> (m, h)
        h_next = np.tanh((concat @ self.Wh) + self.bh)

        # Compute output with softmax activation
        # h_next @ Wy: (m, h) @ (h, o) -> (m, o)
        z = h_next @ self.Wy + self.by

        # Softmax activation
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, y


def rnn(rnn_cell, X, h_0):
    """
    Args:
        rnn_cell is an instance of RNNCell that
        will be used for the forward propagation
        X is the data to be used, given as a
        numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
        h_0 is the initial hidden state, given as a
        numpy.ndarray of shape (m, h)
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
