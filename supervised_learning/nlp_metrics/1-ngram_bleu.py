#!/usr/bin/env python3
"""This module includes the function that
calculates the n-gram BLEU score for a sentence"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Args:
        references is a list of reference translations
        each reference translation is a list of the words in the translation
        sentence is a list containing the model proposed sentence
        n is the size of the n-gram to use for evaluation
    Returns:
        the n-gram BLEU score
    """
    # Generating ngrams from a sentence
    def get_ngram(words, n):
        """N-gram representation of sentence"""
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams.append(ngram)
        return ngrams

    c = len(sentence)
    ref_lengths = [len(ref) for ref in references]
    r = min(ref_lengths, key=lambda ref_length: abs(c - ref_length))

    candidate_ngrams = get_ngram(sentence, n)
    if len(candidate_ngrams) == 0:
        return 0

    candidate_counts = {}
    for word in candidate_ngrams:
        candidate_counts[word] = candidate_counts.get(word, 0) + 1

    clipped_counts = {}
    for ngram in candidate_counts:
        max_ref_count = 0
        for reference in references:
            ref_ngrams = get_ngram(reference, n)
            ref_length = ref_ngrams.count(ngram)
            max_ref_count = max(max_ref_count, ref_length)
        clipped_counts[ngram] = min(max_ref_count, candidate_counts[ngram])

    total_clipped = sum(clipped_counts.values())
    total_ngrams = len(candidate_ngrams)
    precision = total_clipped / total_ngrams

    BP = np.exp(1 - r / c) if c < r else 1

    bleu_score = BP * precision

    return bleu_score
