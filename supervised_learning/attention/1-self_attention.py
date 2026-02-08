#!/usr/bin/env python3
"""Self Attention mechanism for machine translation"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Self Attention class to calculate
       attention for machine translation"""

    def __init__(self, units):
        """
        Initialize the Self Attention layer

        Args:
            units: integer representing the number of hidden units in the
                   alignment model
        """
        super(SelfAttention, self).__init__()

        # Dense layer to be applied to the
        # previous decoder hidden state
        self.W = tf.keras.layers.Dense(units)

        # Dense layer to be applied to the
        # encoder hidden states
        self.U = tf.keras.layers.Dense(units)

        # Dense layer to be applied to the
        # tanh of the sum of W and U outputs
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Calculate the context vector and attention weights

        Args:
            s_prev: tensor of shape (batch, units)
             containing the previous
                   decoder hidden state
            hidden_states: tensor of shape
             (batch, input_seq_len, units)
               containing the outputs of the encoder

        Returns:
            context: tensor of shape (batch, units)
             that contains the context vector for the decoder
            weights: tensor of shape (batch, input_seq_len, 1)
             that contains the attention weights
        """
        # Expand s_prev dimensions to (batch, 1, units) for broadcasting
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # Applying W to s_prev: (batch, 1, units)
        W_s = self.W(s_prev_expanded)

        # Apply U to hidden_states: (batch, input_seq_len, units)
        U_h = self.U(hidden_states)

        # Computing the alignment scores (energy)
        # tanh(W(s_prev) + U(hidden_states)):
        # (batch, input_seq_len, units)
        tanh_sum = tf.nn.tanh(W_s + U_h)

        score = self.V(tanh_sum)

        weights = tf.nn.softmax(score, axis=1)

        # Compute context vector as weighted sum of hidden_states
        # (batch, input_seq_len, units) * (batch, input_seq_len, 1)
        # -> sum over input_seq_len -> (batch, units)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
