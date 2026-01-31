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
    embedding_matrix = model.wv.vectors

    # Get the raw numpy array of embeddings
    # (shape: vocab_size x vector_size)
    vocab_size, embedding_dim = embedding_matrix.shape

    layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=True
    )

    return layer
