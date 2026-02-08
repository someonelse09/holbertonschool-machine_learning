#!/usr/bin/env python3
"""RNN Decoder for machine translation"""
import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN Decoder class for decoding output sequences with attention"""

    def __init__(self, vocab, embedding, units, batch):
        """
        Initialize the RNN Decoder

        Args:
            vocab: integer representing the size of the output vocabulary
            embedding: integer representing the dimensionality of the
                      embedding vector
            units: integer representing the number of hidden units in
                  the RNN cell
            batch: integer representing the batch size
        """
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units

        # Embedding layer to convert word indices to embedding vectors
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        # Self-attention layer
        self.attention = SelfAttention(units)

        # GRU layer with glorot_uniform initialization for recurrent weights
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

        # Dense layer to produce output vocabulary distribution
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Forward pass through the decoder

        Args:
            x: tensor of shape (batch, 1) containing the previous word in
               the target sequence as an index of the target vocabulary
            s_prev: tensor of shape (batch, units) containing the previous
                   decoder hidden state
            hidden_states: tensor of shape (batch, input_seq_len, units)
                          containing the outputs of the encoder

        Returns:
            y: tensor of shape (batch, vocab) containing the output word
               as a one hot vector in the target vocabulary
            s: tensor of shape (batch, units) containing the new decoder
               hidden state
        """
        # Calculate context vector and attention weights
        # context: (batch, units), weights: (batch, input_seq_len, 1)
        context, _ = self.attention(s_prev, hidden_states)

        # Convert word index to embedding: (batch, 1) -> (batch, 1, embedding)
        x = self.embedding(x)

        # Concatenate context vector with embedding
        # Expand context to (batch, 1, units) to match x's sequence dimension
        context_expanded = tf.expand_dims(context, 1)

        # Concatenate along the last dimension: (batch, 1, embedding + units)
        x = tf.concat([context_expanded, x], axis=-1)

        # Pass through GRU layer
        # outputs: (batch, 1, units), s: (batch, units)
        outputs, s = self.gru(x, initial_state=s_prev)

        # Remove the sequence dimension from outputs: (batch, 1, units) -> (batch, units)
        outputs = tf.squeeze(outputs, axis=1)

        # Apply dense layer to get vocabulary distribution: (batch, vocab)
        y = self.F(outputs)

        return y, s
