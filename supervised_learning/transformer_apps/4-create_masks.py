#!/usr/bin/env python3
"""
Function to create masks for Transformer training
"""
import tensorflow as tf


def create_padding_mask(seq):
    """
    Create padding mask for a sequence

    Args:
        seq: tf.Tensor of shape (batch_size, seq_len)

    Returns:
        mask: tf.Tensor of shape (batch_size, 1, 1, seq_len)
    """
    # Create mask where padding (0) becomes 1, others become 0
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # Add extra dimensions for broadcasting
    # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    Create look-ahead mask to mask future tokens

    Args:
        size: int, size of the sequence

    Returns:
        mask: tf.Tensor of shape (size, size)
    """
    # Create upper triangular matrix of 1s
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks(inputs, target):
    """
    Create all masks for training/validation

    Args:
        inputs: tf.Tensor of shape (batch_size, seq_len_in)
                contains the input sentence
        target: tf.Tensor of shape (batch_size, seq_len_out)
                contains the target sentence

    Returns:
        encoder_mask: tf.Tensor padding mask of shape
                      (batch_size, 1, 1, seq_len_in)
                      to be applied in the encoder
        combined_mask: tf.Tensor of shape
                       (batch_size, 1, seq_len_out, seq_len_out)
                       used in the 1st attention block in the decoder
        decoder_mask: tf.Tensor padding mask of shape
                      (batch_size, 1, 1, seq_len_in)
                      used in the 2nd attention block in the decoder
    """
    encoder_mask = create_padding_mask(inputs)

    # Decoder padding mask (used in 2nd attention block)
    decoder_mask = create_padding_mask(inputs)

    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = create_look_ahead_mask(seq_len_out)

    target_padding_mask = create_padding_mask(target)

    # Combined mask: maximum of look-ahead and padding masks
    combined_mask = tf.maximum(target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
