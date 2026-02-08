#!/usr/bin/env python3
"""Transformer Encoder"""
import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Encoder for a Transformer"""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Initialize the Encoder

        Args:
            N: the number of blocks in the encoder
            dm: the dimensionality of the model
            h: the number of heads
            hidden: the number of hidden units
             in the fully connected layer
            input_vocab: the size of the input vocabulary
            max_seq_len: the maximum sequence length possible
            drop_rate: the dropout rate
        """
        super(Encoder, self).__init__()

        self.N = N
        self.dm = dm

        # Embedding layer for the inputs
        self.embedding = tf.keras.layers.Embedding(
            input_dim=input_vocab,
            output_dim=dm
        )

        # Positional encoding
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # Create N encoder blocks
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        # Dropout layer for positional encodings
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training=False, mask=None):
        """
        Forward pass through the encoder

        Args:
            x: tensor of shape (batch, input_seq_len)
             containing the input to the encoder (word indices)
            training: boolean to determine if the model is training
            mask: the mask to be applied for multi head attention

        Returns:
            A tensor of shape (batch, input_seq_len, dm)
             containing the encoder output
        """
        # Get sequence length
        seq_len = x.shape[1]

        # Convert input indices to embeddings
        # x: (batch, input_seq_len) -> (batch, input_seq_len, dm)
        x = self.embedding(x)

        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encoding
        # positional_encoding: (max_seq_len, dm)
        # We only use the first seq_len positions
        x += self.positional_encoding[:seq_len, :]

        x = self.dropout(x, training=training)

        # Pass through all N encoder blocks
        for block in self.blocks:
            x = block(x, training=training, mask=mask)

        return x
