#!/usr/bin/env python3
"""RNN Encoder for machine translation"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNN Encoder class for encoding input sequences"""

    def __init__(self, vocab, embedding, units, batch):
        """
        Initialize the RNN Encoder

        Args:
            vocab: integer representing the size of the input vocabulary
            embedding: integer representing the dimensionality of the
                      embedding vector
            units: integer representing the number of hidden units in
                  the RNN cell
            batch: integer representing the batch size
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units

        # Embedding layer to convert word indices to embedding vectors
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        # GRU layer with glorot_uniform initialization for recurrent weights
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of zeros

        Returns:
            A tensor of shape (batch, units) containing the initialized
            hidden states
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Forward pass through the encoder

        Args:
            x: tensor of shape (batch, input_seq_len) containing the input
               to the encoder as word indices within the vocabulary
            initial: tensor of shape (batch, units) containing the initial
                    hidden state

        Returns:
            outputs: tensor of shape (batch, input_seq_len, units) containing
                    the outputs of the encoder
            hidden: tensor of shape (batch, units) containing the last hidden
                   state of the encoder
        """
        # Convert word indices to embeddings
        x = self.embedding(x)

        # Pass through GRU layer
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
