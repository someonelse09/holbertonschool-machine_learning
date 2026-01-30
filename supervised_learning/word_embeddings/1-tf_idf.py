#!/usr/bin/env python3
"""This module includes the function
that creates a TF-IDF embedding"""
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use for analysis
               If None, all words within sentences should be used

    Returns:
        embeddings: numpy.ndarray of shape
         (s, f) containing the embeddings
                   s is the number of sentences
                   f is the number of features analyzed
        features: numpy array of the features used for embeddings
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

    # Create feature list (vocabulary)
    features = np.array(vocab)

    # Create word to index mapping
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    # Initialize matrices
    s = len(sentences)
    f = len(vocab)

    # Step 1: Calculate Term Frequency (TF)
    tf_matrix = np.zeros((s, f), dtype=float)

    for i, words in enumerate(tokenized_sentences):
        # Count words in this sentence
        word_count = {}
        for word in words:
            if word in word_to_idx:
                word_count[word] = word_count.get(word, 0) + 1

        # Calculate TF for each word in vocabulary
        total_words = len(words) if words else 1
        for word, count in word_count.items():
            idx = word_to_idx[word]
            tf_matrix[i, idx] = count / total_words

    # Step 2: Calculate Inverse Document Frequency (IDF)
    idf_vector = np.zeros(f, dtype=float)

    for j, word in enumerate(vocab):
        # Count number of documents containing this word
        doc_count = 0
        for words in tokenized_sentences:
            if word in words:
                doc_count += 1

        # Calculate IDF
        if doc_count > 0:
            idf_vector[j] = np.log((s + 1) / (doc_count + 1)) + 1
        else:
            idf_vector[j] = 1

    # Step 3: Calculate TF-IDF
    tfidf_matrix = tf_matrix * idf_vector

    # Step 4: L2 Normalization (normalize each row)
    for i in range(s):
        norm = np.linalg.norm(tfidf_matrix[i])
        if norm > 0:
            tfidf_matrix[i] = tfidf_matrix[i] / norm

    return tfidf_matrix, features
