#!/usr/bin/env python3
"""This class includes the class Dataset that
loads and prepares a dataset for machine translation"""
import tensorflow_datasets as tfds
from transformers import AutoTokenizer
import numpy as np

class Dataset:
    """
    Dataset class for machine translation (Portuguese to English)
    Loads the TED HRLR translation dataset and creates tokenizers
    """
    def __init__(self):
        """
        Args:
            data_train, which contains the ted_hrlr_translate/
             pt_to_en tf.data.Dataset train split, loaded as_supervised
            data_valid, which contains the ted_hrlr_translate/
             pt_to_en tf.data.Dataset validate split, loaded as_supervised
            tokenizer_pt is the Portuguese tokenizer created from the training set
            tokenizer_en is the English tokenizer created from the training set
        """
        # as_supervised=True returns (input, label) tuples
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for our dataset
        Args:
            data is a tf.data.Dataset whose examples
             are formatted as a tuple (pt, en)
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the
             corresponding English sentence
            Use a pre-trained tokenizer:
            use the pretrained model neuralmind/
             bert-base-portuguese-cased for the portuguese text
            use the pretrained model bert-base-uncased for the english text
            Train the tokenizers with a maximum vocabulary size of 2**13
        Returns:
            tokenizer_pt, tokenizer_en
            tokenizer_pt is the Portuguese tokenizer
            tokenizer_en is the English tokenizer
        """
        tokenizer_pt = AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )
        pt_sentences = []
        en_sentences = []

        # Extract sentences from the dataset
        # Decode byte strings to UTF-8 text
        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))
        vocab_size = 2 ** 13

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences,
            vocab_size=vocab_size
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences,
            vocab_size=vocab_size
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encodes a translation into tokens
        Args:
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the corresponding English sentence
            The tokenized sentences should include
             the start and end of sentence tokens
            The start token should be indexed as vocab_size
            The end token should be indexed as vocab_size + 1
        Returns:
            pt_tokens, en_tokens
            pt_tokens is a np.ndarray containing the Portuguese tokens
            en_tokens is a np.ndarray. containing the English tokens
        """
        vocab_size = self.tokenizer_pt.vocab_size

        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_tokens = self.tokenizer_pt.encode(pt_text)
        en_tokens = self.tokenizer_en.encode(en_text)

        pt_tokens = [vocab_size] + pt_tokens + [vocab_size + 1]
        en_tokens = [vocab_size] + en_tokens +  [vocab_size + 1]

        pt_tokens = np.array(pt_tokens)
        en_tokens = np.array(en_tokens)

        return pt_tokens, en_tokens
