#!/usr/bin/env python3
"""This module is composed of the function that
calculates the cumulative n-gram BLEU score for a sentence"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Args:
        references is a list of reference translations
        each reference translation is a list of the words in the translation
        sentence is a list containing the model proposed sentence
        n is the size of the largest n-gram to use for evaluation
        All n-gram scores should be weighted evenly
    Returns:
        the cumulative n-gram BLEU score
    """
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
    log_sum = 0

    for i in range(1, n + 1):
        candidate_ngrams = get_ngram(sentence, i)
        if len(candidate_ngrams) == 0:
            # Can't compute this n-gram, but don't fail entirely
            # Just use a very small value to avoid log(0)
            log_sum += (1 / n) * np.log(1e-10)
            continue

        # Counting candidate n-grams
        ngram_counts = {}
        for ngram in candidate_ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        clipped_ngrams = {}
        for ngram in ngram_counts:
            max_ref_len = 0
            for reference in references:
                ref_ngrams = get_ngram(reference, i)
                ref_len = ref_ngrams.count(ngram)
                max_ref_len = max(max_ref_len, ref_len)
            clipped_ngrams[ngram] = min(max_ref_len, ngram_counts[ngram])
        total_ngram_sum = sum(clipped_ngrams.values())
        total_sentence_sum = len(candidate_ngrams)
        pi = total_ngram_sum / total_sentence_sum
        log_sum += (1 / n) * np.log(pi)

    BP = np.exp(1 - r / c) if r > c else 1
    bleu_score = BP * np.exp(log_sum)

    return bleu_score
