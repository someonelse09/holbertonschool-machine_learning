#!/usr/bin/env python3
"""Positional Encoding for Transformer"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculate the positional encoding for a transformer

    Args:
        max_seq_len: integer representing the maximum sequence length
        dm: the model depth (dimensionality)

    Returns:
        A numpy.ndarray of shape (max_seq_len, dm) containing the
        positional encoding vectors
    """
    # Initialize the positional encoding matrix
    PE = np.zeros((max_seq_len, dm))

    # Create position indices: [0, 1, 2, ..., max_seq_len-1]
    # Shape: (max_seq_len, 1)
    position = np.arange(max_seq_len)[:, np.newaxis]

    # Create dimension indices: [0, 2, 4, ..., dm-2]
    # We use steps of 2 because we apply sin to even indices and cos to odd
    # Shape: (dm // 2,)
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))

    # Apply sine to even indices in the array (0, 2, 4, ...)
    PE[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd indices in the array (1, 3, 5, ...)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE
