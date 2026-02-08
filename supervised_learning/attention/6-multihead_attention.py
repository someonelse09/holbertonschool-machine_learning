#!/usr/bin/env python3
"""Multi Head Attention for Transformer"""
import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi Head Attention class"""

    def __init__(self, dm, h):
        """
        Initialize Multi Head Attention.

        Args:
            dm: integer representing the dimensionality of the model
            h: integer representing the number of heads
            dm is divisible by h
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        # Dense layers to generate Q, K, V matrices
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        # Dense layer for final linear transformation
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (h, depth).
        Transpose the result to shape (batch, h, seq_len, depth).

        Args:
            x: tensor of shape (batch, seq_len, dm)
            batch_size: batch size

        Returns:
            Tensor of shape (batch, h, seq_len, depth)
        """
        # Reshape from (batch, seq_len, dm) to
        # (batch, seq_len, h, depth)
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))

        # Transpose to (batch, h, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Forward pass through multi-head attention.

        Args:
            Q: tensor of shape (batch, seq_len_q, dk) containing
               input to generate the query matrix
            K: tensor of shape (batch, seq_len_v, dk) containing
               input to generate the key matrix
            V: tensor of shape (batch, seq_len_v, dv) containing
               input to generate the value matrix
            mask: always None

        Returns:
            output: tensor with shape (..., seq_len_q, dm)
                    containing the scaled dot product attention
            weights: tensor with shape
                     (..., h, seq_len_q, seq_len_v) containing
                     the attention weights
        """
        batch_size = tf.shape(Q)[0]

        # Generate Q, K, V matrices through dense layers
        # Q, K, V: (batch, seq_len, dm)
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        # Split into multiple heads
        # Q: (batch, h, seq_len_q, depth)
        # K: (batch, h, seq_len_v, depth)
        # V: (batch, h, seq_len_v, depth)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Apply scaled dot product attention
        # scaled_attention: (batch, h, seq_len_q, depth)
        # weights: (batch, h, seq_len_q, seq_len_v)
        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        # Transpose back to
        # (batch, seq_len_q, h, depth)
        scaled_attention = tf.transpose(
            scaled_attention,
            perm=[0, 2, 1, 3]
        )

        # Concatenate heads:
        # (batch, seq_len_q, h, depth) -> (batch, seq_len_q, dm)
        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.dm)
        )

        # Apply final linear layer
        output = self.linear(concat_attention)

        return output, weights
