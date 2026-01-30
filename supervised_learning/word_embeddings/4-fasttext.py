#!/usr/bin/env python3
"""This module includes the function that
creates, builds and trains a gensim fastText model"""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5,
                   negative=5, window=5, cbow=True, epochs=5,
                   seed=0, workers=1):
    """
    Args:
        sentences is a list of sentences to be trained on
        vector_size is the dimensionality of the embedding layer
        min_count is the minimum number of occurrences
         of a word for use in training
        window is the maximum distance between the
         current and predicted word within a sentence
        negative is the size of negative sampling
        cbow is a boolean to determine the training type;
         True is for CBOW; False is for Skip-gram
        epochs is the number of iterations to train over
        seed is the seed for the random number generator
        workers is the number of worker threads to train the model
    Returns:
        the trained model
    """
    sg = 0 if cbow else 1

    model = gensim.models.FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
