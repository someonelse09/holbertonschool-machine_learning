#!/usr/bin/env python3
"""
Module containing the gensim_to_keras function
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer.

    Args:
        model: A trained gensim word2vec model

    Returns:
        The trainable keras Embedding layer
    """
    # 1. Extract the weights matrix from the Gensim model
    # model.wv (KeyedVectors) holds the mapping between words and vectors
    keyed_vectors = model.wv
    weights = keyed_vectors.vectors  # Shape: (vocab_size, vector_size)

    # 2. Create the Keras Embedding Layer
    # We initialize it with the weights we extracted.
    layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],    # Vocabulary size
        output_dim=weights.shape[1],   # Vector size
        weights=[weights],             # Keras expects a list of arrays
        trainable=True                 # Allow fine-tuning
    )

    return layer
