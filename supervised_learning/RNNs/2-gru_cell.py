#!/usr/bin/env python3
"""This module includes the class GRUCell
that represents a gated recurrent unit"""
import numpy as np


class GRUCell:
    """Single Gated Recurrent Unit"""
    def __init__(self, i, h, o):
        """
        Args:
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs
            Creates the public instance attributes Wz, Wr, Wh, Wy,
             bz, br, bh, by that represent the weights and biases of the cell
            Wz and bz are for the update gate
            Wr and br are for the reset gate
            Wh and bh are for the intermediate hidden state
            Wy and by are for the output
            The weights should be initialized using a
             random normal distribution in the order listed above
            The weights will be used on the right side for matrix multiplication
            The biases should be initialized as zeros
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ Performs forward propagation for one time step
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
        # Concatenate h_prev and x_t: shape (m, h) + (m, i) -> (m, h+i)
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Reset gate: determines how much of the previous hidden state to forget
        # sigmoid(concat @ Wr + br): (m, h+i) @ (h+i, h) -> (m, h)
        r = 1 / (1 + np.exp(-(concat @ self.Wr + self.br)))

        # Update gate: determines how much of the new candidate to use
        # sigmoid(concat @ Wz + bz): (m, h+i) @ (h+i, h) -> (m, h)
        z = 1 / (1 + np.exp(-(concat @ self.Wz + self.bz)))

        # Candidate hidden state with reset gate applied element-wise to h_prev
        # Concatenate (r * h_prev) and x_t: (m, h) + (m, i) -> (m, h+i)
        concat_reset = np.concatenate((r * h_prev, x_t), axis=1)

        # Compute candidate: tanh(concat_reset @ Wh + bh)
        # (m, h+i) @ (h+i, h) -> (m, h)
        candidate_ht = np.tanh(concat_reset @ self.Wh + self.bh)

        # Compute next hidden state using element-wise operations
        # h_next = (1 - z) * h_prev + z * h_candidate
        # All shapes: (m, h)
        h_next = (1 - z) * h_prev + z * candidate_ht

        # Compute output with softmax activation
        # h_next @ Wy: (m, h) @ (h, o) -> (m, o)
        logits = h_next @ self.Wy + self.by
        exp_z = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, y
