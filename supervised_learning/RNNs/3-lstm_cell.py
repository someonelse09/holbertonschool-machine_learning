#!/usr/bin/env python3
"""This module includes the class
LSTMCell that represents an LSTM unit"""
import numpy as np


class LSTMCell:
    """A single cell of Long Short Term Memory"""
    def __init__(self, i, h, o):
        """
        Args:
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs
            Creates the public instance attributes Wf, Wu, Wc, Wo, Wy,
             bf, bu, bc, bo, by that represent the weights and biases of the cell
            Wf and bf are for the forget gate
            Wu and bu are for the update gate
            Wc and bc are for the intermediate cell state
            Wo and bo are for the output gate
            Wy and by are for the outputs
            The weights should be initialized using
             a random normal distribution in the order listed above
            The weights will be used on the right side for matrix multiplication
            The biases should be initialized as zeros
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one time step
        Args:
            x_t is a numpy.ndarray of shape (m, i) that contains the data input for the cell
            m is the batche size for the data
            h_prev is a numpy.ndarray of shape (m, h) containing the previous hidden state
            c_prev is a numpy.ndarray of shape (m, h) containing the previous cell state
            The output of the cell should use a softmax activation function
        Returns: h_next, c_next, y
            h_next is the next hidden state
            c_next is the next cell state
            y is the output of the cell
        """
        # Concatenate h_prev and x_t: shape (m, h) + (m, i) -> (m, h+i)
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate: decides what information to discard from cell state
        # sigmoid(concat @ Wf + bf): (m, h+i) @ (h+i, h) -> (m, h)
        f_t = 1 / (1 + np.exp(-(concat @ self.Wf + self.bf)))

        # Input/Update gate: decides what new information to store in cell state
        # sigmoid(concat @ Wu + bu): (m, h+i) @ (h+i, h) -> (m, h)
        i_t = 1 / (1 + np.exp(-(concat @ self.Wu + self.bu)))

        # Candidate cell state: new information to potentially add
        # tanh(concat @ Wc + bc): (m, h+i) @ (h+i, h) -> (m, h)
        c_hat = np.tanh(concat @ self.Wc + self.bc)

        # Update cell state: forget old info + add new info (element-wise operations)
        # c_next = f_t * c_prev + i_t * c_hat (all shapes: (m, h))
        c_next = f_t * c_prev + i_t * c_hat

        # Output gate: decides what parts of cell state to output
        # sigmoid(concat @ Wo + bo): (m, h+i) @ (h+i, h) -> (m, h)
        o_t = 1 / (1 + np.exp(-(concat @ self.Wo + self.bo)))

        # Compute next hidden state (element-wise multiplication)
        # h_next = o_t * tanh(c_next) (all shapes: (m, h))
        h_next = o_t * np.tanh(c_next)

        # Compute output with softmax activation
        # h_next @ Wy: (m, h) @ (h, o) -> (m, o)
        z = h_next @ self.Wy + self.by
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, c_next, y
