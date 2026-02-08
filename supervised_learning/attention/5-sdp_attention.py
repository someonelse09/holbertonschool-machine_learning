#!/usr/bin/env python3
"""Scaled Dot Product Attention"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculate the scaled dot product attention.

    Args:
        Q: tensor with its last two dimensions as
           (..., seq_len_q, dk) containing the query matrix
        K: tensor with its last two dimensions as
           (..., seq_len_v, dk) containing the key matrix
        V: tensor with its last two dimensions as
           (..., seq_len_v, dv) containing the value matrix
        mask: tensor that can be broadcast into
              (..., seq_len_q, seq_len_v) containing the optional
              mask, or defaulted to None

    Returns:
        output: tensor with its last two dimensions as
                (..., seq_len_q, dv) containing the scaled dot
                product attention
        weights: tensor with its last two dimensions as
                 (..., seq_len_q, seq_len_v) containing the
                 attention weights
    """
    # Calculate Q * K^T
    # Q: (..., seq_len_q, dk)
    # K^T: (..., dk, seq_len_v)
    # matmul: (..., seq_len_q, seq_len_v)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Get the depth (dk) for scaling
    dk = tf.cast(tf.shape(K)[-1], tf.float32)

    # Scale the matmul by sqrt(dk)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Apply mask if provided
    if mask is not None:
        # Add -1e9 (negative infinity) to masked positions
        scaled_attention_logits += mask * -1e9

    # Apply softmax to get attention weights
    # softmax is applied on the last axis (seq_len_v)
    # weights: (..., seq_len_q, seq_len_v)
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiply weights by V
    # weights: (..., seq_len_q, seq_len_v)
    # V: (..., seq_len_v, dv)
    # output: (..., seq_len_q, dv)
    output = tf.matmul(weights, V)

    return output, weights
