#!/usr/bin/env python3
"""
Transformer model for machine translation
"""
import tensorflow as tf
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculate positional encoding for a transformer

    Args:
        max_seq_len: maximum sequence length
        dm: model depth

    Returns:
        positional encoding as numpy array of shape (max_seq_len, dm)
    """
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, dm, 2) * -(np.log(10000.0) / dm)
    )

    pos_encoding = np.zeros((max_seq_len, dm))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return pos_encoding


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention layer
    """

    def __init__(self, dm, h):
        """
        Initialize multi-head attention

        Args:
            dm: dimensionality of the model
            h: number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (h, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        """
        Forward pass for multi-head attention

        Args:
            q: query tensor
            k: key tensor
            v: value tensor
            mask: mask to apply

        Returns:
            output, attention_weights
        """
        batch_size = tf.shape(q)[0]

        # Linear projections
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Add mask
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Softmax
        attention_weights = tf.nn.softmax(
            scaled_attention_logits, axis=-1
        )

        # Apply attention to values
        output = tf.matmul(attention_weights, v)

        # Concatenate heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.dm))

        # Final linear projection
        output = self.linear(output)

        return output, attention_weights


class EncoderBlock(tf.keras.layers.Layer):
    """
    Encoder block for transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialize encoder block

        Args:
            dm: dimensionality of the model
            h: number of heads
            hidden: number of hidden units in feed forward
            drop_rate: dropout rate
        """
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='relu'),
            tf.keras.layers.Dense(dm)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Forward pass for encoder block

        Args:
            x: input tensor
            training: boolean for training mode
            mask: mask to apply

        Returns:
            output tensor
        """
        # Multi-head attention
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output = self.layernorm2(out1 + ffn_output)

        return output


class DecoderBlock(tf.keras.layers.Layer):
    """
    Decoder block for transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialize decoder block

        Args:
            dm: dimensionality of the model
            h: number of heads
            hidden: number of hidden units in feed forward
            drop_rate: dropout rate
        """
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='relu'),
            tf.keras.layers.Dense(dm)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_output, training, look_ahead_mask,
             padding_mask):
        """
        Forward pass for decoder block

        Args:
            x: input tensor
            enc_output: encoder output
            training: boolean for training mode
            look_ahead_mask: look ahead mask
            padding_mask: padding mask

        Returns:
            output tensor
        """
        # Masked multi-head attention
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # Multi-head attention with encoder output
        attn2, _ = self.mha2(
            out1, enc_output, enc_output, padding_mask
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        # Feed forward network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        output = self.layernorm3(ffn_output + out2)

        return output


class Encoder(tf.keras.layers.Layer):
    """
    Encoder for transformer
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Initialize encoder

        Args:
            N: number of blocks
            dm: dimensionality of the model
            h: number of heads
            hidden: number of hidden units
            input_vocab: input vocabulary size
            max_seq_len: maximum sequence length
            drop_rate: dropout rate
        """
        super(Encoder, self).__init__()

        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(
            input_vocab, dm
        )
        self.positional_encoding = positional_encoding(
            max_seq_len, dm
        )

        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate)
            for _ in range(N)
        ]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Forward pass for encoder

        Args:
            x: input tensor
            training: boolean for training mode
            mask: mask to apply

        Returns:
            output tensor
        """
        seq_len = tf.shape(x)[1]

        # Embedding and positional encoding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]

        x = self.dropout(x, training=training)

        # Pass through encoder blocks
        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """
    Decoder for transformer
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Initialize decoder

        Args:
            N: number of blocks
            dm: dimensionality of the model
            h: number of heads
            hidden: number of hidden units
            target_vocab: target vocabulary size
            max_seq_len: maximum sequence length
            drop_rate: dropout rate
        """
        super(Decoder, self).__init__()

        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(
            target_vocab, dm
        )
        self.positional_encoding = positional_encoding(
            max_seq_len, dm
        )

        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate)
            for _ in range(N)
        ]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_output, training, look_ahead_mask,
             padding_mask):
        """
        Forward pass for decoder

        Args:
            x: input tensor
            enc_output: encoder output
            training: boolean for training mode
            look_ahead_mask: look ahead mask
            padding_mask: padding mask

        Returns:
            output tensor
        """
        seq_len = tf.shape(x)[1]

        # Embedding and positional encoding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]

        x = self.dropout(x, training=training)

        # Pass through decoder blocks
        for i in range(self.N):
            x = self.blocks[i](
                x, enc_output, training, look_ahead_mask,
                padding_mask
            )

        return x


class Transformer(tf.keras.Model):
    """
    Transformer model for machine translation
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Initialize transformer

        Args:
            N: number of blocks
            dm: dimensionality of the model
            h: number of heads
            hidden: number of hidden units
            input_vocab: input vocabulary size
            target_vocab: target vocabulary size
            max_seq_input: maximum input sequence length
            max_seq_target: maximum target sequence length
            drop_rate: dropout rate
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate
        )

        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target,
            drop_rate
        )

        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        Forward pass for transformer

        Args:
            inputs: input tensor
            target: target tensor
            training: boolean for training mode
            encoder_mask: encoder mask
            look_ahead_mask: look ahead mask
            decoder_mask: decoder mask

        Returns:
            output tensor
        """
        # Encoder
        enc_output = self.encoder(inputs, training, encoder_mask)

        # Decoder
        dec_output = self.decoder(
            target, enc_output, training, look_ahead_mask,
            decoder_mask
        )

        # Final linear layer
        output = self.linear(dec_output)

        return output
