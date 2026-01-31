#!/usr/bin/env python3
"""This module includes the funciton that
converts a gensim word2vec model to a keras Embedding layer"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer

    Args:
        model: trained gensim word2vec model

    Returns:
        Trainable keras Embedding layer with weights from the word2vec
        model
    """
    # Get the word vectors from the model
    word_vectors = model.wv

    # Get the vocabulary size and vector dimensions
    vocab_size = len(word_vectors)
    vector_dim = word_vectors.vector_size

    # Get the weight matrix (word vectors)
    # word_vectors.vectors contains all the word embeddings
    weights = word_vectors.vectors

    # Create a Keras Embedding layer
    # input_dim: size of vocabulary
    # output_dim: dimension of embeddings
    # weights: initial weights (from word2vec)
    # trainable: True (weights can be further updated)
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_dim,
        weights=[weights],
        trainable=True
    )

    return embedding_layer
