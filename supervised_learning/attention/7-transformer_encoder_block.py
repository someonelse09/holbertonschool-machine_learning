#!/usr/bin/env python3
"""Transformer Encoder Block"""
import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Encoder Block for a Transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialize the Encoder Block.

        Args:
            dm: the dimensionality of the model
            h: the number of heads
            hidden: the number of hidden units in the fully connected layer
            drop_rate: the dropout rate
        """
        super(EncoderBlock, self).__init__()

        # Multi-head attention layer
        self.mha = MultiHeadAttention(dm, h)

        # Feed-forward network with two dense layers
        self.dense_hidden = tf.keras.layers.Dense(
            hidden,
            activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(dm)

        # Layer normalization layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Forward pass through the encoder block.

        Args:
            x: tensor of shape (batch, input_seq_len, dm)
               containing the input to the encoder block
            training: boolean to determine if the model is training
            mask: the mask to be applied for multi head attention

        Returns:
            Tensor of shape (batch, input_seq_len, dm)
            containing the block's output
        """
        # Multi-head attention
        attn_output, _ = self.mha(x, x, x, mask)

        # Apply dropout
        attn_output = self.dropout1(
            attn_output,
            training=training
        )

        # Add & Norm
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward network
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)

        # Apply dropout
        ffn_output = self.dropout2(
            ffn_output,
            training=training
        )

        # Add & Norm
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
