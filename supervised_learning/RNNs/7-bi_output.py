#!/usr/bin/env python3
"""Module for bidirectional RNN cell"""
import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of an RNN"""

    def __init__(self, i, h, o):
        """Constructor

        Args:
            i: the dimensionality of the data
            h: the dimensionality of the hidden states
            o: the dimensionality of the outputs

        Creates the public instance attributes Whf, Whb, Wy, bhf,
        bhb, by that represent the weights and biases of the cell
        - Whf and bhf are for the hidden states in the forward
          direction
        - Whb and bhb are for the hidden states in the backward
          direction
        - Wy and by are for the outputs

        The weights should be initialized using a random normal
        distribution in the order listed above.
        The weights will be used on the right side for matrix
        multiplication.
        The biases should be initialized as zeros.
        """
        # Initialize weights in order: Whf, Whb, Wy
        # Forward direction weights
        self.Whf = np.random.randn(i + h, h)

        # Backward direction weights
        self.Whb = np.random.randn(i + h, h)

        # Output weights (takes concatenated forward and backward
        # hidden states)
        self.Wy = np.random.randn(2 * h, o)

        # Initialize biases in order: bhf, bhb, by
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Calculates the hidden state in the forward direction
        for one time step

        Args:
            x_t: numpy.ndarray of shape (m, i) that contains the
                 data input for the cell
                m is the batch size for the data
            h_prev: numpy.ndarray of shape (m, h) containing the
                    previous hidden state

        Returns:
            h_next: the next hidden state
        """
        # Concatenate h_prev and x_t: (m, h) + (m, i) -> (m, h+i)
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute next hidden state with tanh activation
        # concat @ Whf: (m, h+i) @ (h+i, h) -> (m, h)
        h_next = np.tanh(concat @ self.Whf + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """Calculates the hidden state in the backward direction
         for one time step

        Args:
            x_t is a numpy.ndarray of shape (m, i)
             that contains the data input for the cell
            m is the batch size for the data
            h_next is a numpy.ndarray of shape (m, h)
             containing the next hidden state
        Returns:
            h_pev, the previous hidden state
        """
        # Concatenate h_next and x_t: (m, h) + (m, i) -> (m, h+i)
        concat = np.concatenate((h_next, x_t), axis=1)

        # Compute previous hidden state with tanh activation
        # concat @ Whb: (m, h+i) @ (h+i, h) -> (m, h)
        h_prev = np.tanh(concat @ self.Whb + self.bhb)

        return h_prev

    def output(self, H):
        """Calculates all outputs for the RNN
        Args:
            H is a numpy.ndarray of shape (t, m, 2 * h) that
             contains the concatenated hidden states from both directions,
              excluding their initialized states
            t is the number of time steps
            m is the batch size for the data
            h is the dimensionality of the hidden states
        Returns:
            Y, the outputs"""
        t, m, _ = H.shape
        Y = []

        for step in range(t):
            # Get concatenated hidden states at this time step
            h_concat = H[step]  # Shape: (m, 2*h)

            # Compute output: h_concat @ Wy + by
            # (m, 2*h) @ (2*h, o) -> (m, o)
            z = h_concat @ self.Wy + self.by

            # Applying softmax activation
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            y = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            Y.append(y)
        Y = np.array(Y)

        return Y
