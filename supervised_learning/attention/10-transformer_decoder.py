#!/usr/bin/env python3
"""Transformer Decoder"""
import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Decoder for a Transformer"""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Initialize the Decoder

        Args:
            N: the number of blocks in the decoder
            dm: the dimensionality of the model
            h: the number of heads
            hidden: the number of hidden units in the fully connected layer
            target_vocab: the size of the target vocabulary
            max_seq_len: the maximum sequence length possible
            drop_rate: the dropout rate
        """
        super(Decoder, self).__init__()

        self.N = N
        self.dm = dm

        # Embedding layer for the target
        self.embedding = tf.keras.layers.Embedding(
            input_dim=target_vocab,
            output_dim=dm
        )

        # Positional encoding
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # Create N decoder blocks
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        # Dropout layer for positional encodings
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training=False, look_ahead_mask=None,
             padding_mask=None):
        """
        Forward pass through the decoder

        Args:
            x: tensor of shape (batch, target_seq_len) containing the input
               to the decoder (target word indices)
            encoder_output: tensor of shape (batch, input_seq_len, dm)
                           containing the output of the encoder
            training: boolean to determine if the model is training
            look_ahead_mask: mask to be applied to the first multi head
                            attention layer
            padding_mask: mask to be applied to the second multi head
                         attention layer

        Returns:
            A tensor of shape (batch, target_seq_len, dm) containing the
            decoder output
        """
        # Get sequence length
        seq_len = x.shape[1]

        # Convert target indices to embeddings
        # x: (batch, target_seq_len) -> (batch, target_seq_len, dm)
        x = self.embedding(x)

        # Scale embeddings by sqrt(dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encoding
        # positional_encoding: (max_seq_len, dm)
        # We only use the first seq_len positions
        x += self.positional_encoding[:seq_len, :]

        # Apply dropout to the sum of embeddings and positional encodings
        x = self.dropout(x, training=training)

        # Pass through all N decoder blocks
        for block in self.blocks:
            x = block(x, encoder_output, training=training,
                      look_ahead_mask=look_ahead_mask,
                      padding_mask=padding_mask)

        return x
