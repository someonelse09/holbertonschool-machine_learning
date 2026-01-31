#!/usr/bin/env python3
"""
Module to convert a gensim Word2Vec
model to a Keras Embedding layer
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
    # Access the KeyedVectors object
    # which holds the learned vectors
    keyed_vectors = model.wv

    # Get the raw numpy array of embeddings
    # (shape: vocab_size x vector_size)
    weights = keyed_vectors.vectors

    # Create the Keras Embedding layer
    # input_dim is the size of the vocabulary
    # output_dim is the dimensionality of the embeddings
    # weights must be passed as a list of numpy arrays
    layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )

    return layer
