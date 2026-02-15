#!/usr/bin/env python3
"""
Dataset class for machine translation
"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Dataset class for loading and preparing machine translation data
    """

    def __init__(self):
        """
        Initialize Dataset with train/validation splits and tokenizers
        Creates instance attributes:
            data_train: training dataset split
            data_valid: validation dataset split
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Create sub-word tokenizers for the dataset

        Args:
            data: tf.data.Dataset with examples as tuple (pt, en)
                pt: tf.Tensor containing Portuguese sentence
                en: tf.Tensor containing English sentence

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        # Load pre-trained tokenizers
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        # Collect sentences for training tokenizers
        pt_sentences = []
        en_sentences = []

        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))

        # Maximum vocabulary size
        vocab_size = 2 ** 13

        # Train new tokenizers
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences,
            vocab_size=vocab_size
        )

        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences,
            vocab_size=vocab_size
        )

        return tokenizer_pt, tokenizer_en
