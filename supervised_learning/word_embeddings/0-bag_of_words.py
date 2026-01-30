#!/usr/bin/env python3
"""This module includes the function for
creating bag of words embeddings"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use for analysis
               If None, all words within sentences should be used

    Returns:
        embeddings: numpy.ndarray of shape
         (s, f) containing the embeddings
                   s is the number of sentences
                   f is the number of features analyzed
        features: list of the features used for embeddings
    """
    # Preprocess sentences: lowercase and tokenize
    tokenized_sentences = []
    for sentence in sentences:
        # Convert to lowercase
        cleaned = sentence.lower()
        # Remove punctuation by replacing it with spaces
        for char in "!.,?;:\"":
            cleaned = cleaned.replace(char, " ")
        # Split into words
        words = cleaned.split()
        # Handle possessives - remove 's suffix
        processed_words = []
        for word in words:
            if word.endswith("'s"):
                word = word[:-2]
            elif word.endswith("'"):
                word = word[:-1]
            if word:  # Only add non-empty words
                processed_words.append(word)
        tokenized_sentences.append(processed_words)

    # Build vocabulary if not provided
    if vocab is None:
        # Collect all unique words from all sentences
        vocab_set = set()
        for words in tokenized_sentences:
            vocab_set.update(words)
        # Sort vocabulary alphabetically
        vocab = sorted(list(vocab_set))
    else:
        # If vocab is provided, use it as is
        vocab = list(vocab)

    # Create feature list (sorted vocabulary)
    features = vocab

    # Create word to index mapping
    word_to_idx = {word: idx for idx, word in enumerate(features)}

    # Initialize embeddings matrix
    s = len(sentences)  # number of sentences
    f = len(features)  # number of features
    embeddings = np.zeros((s, f), dtype=int)

    # Fill embeddings matrix with word counts
    for i, words in enumerate(tokenized_sentences):
        for word in words:
            if word in word_to_idx:
                embeddings[i, word_to_idx[word]] += 1

    return embeddings, np.array(features)
