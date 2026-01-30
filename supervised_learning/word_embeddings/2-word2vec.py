#!/usr/bin/env python3
"""This module inlcudes the function that
creates , builds and trains a gensim word2vec model"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True, epochs=5,
                   seed=0, workers=1):
    """
    Args:
        sentences is a list of sentences to be trained on
        vector_size is the dimensionality of the embedding layer
        min_count is the minimum number of
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
    # Determine the training algorithm
    # sg=0 for CBOW (Continuous Bag of Words)
    # sg=1 for Skip-gram
    sg = 0 if cbow else 1

    # Creating the word to vector model
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )

    return model
