#!/usr/bin/env python3
"""This class includes the class Dataset that
loads and prepares a dataset for machine translation"""
import tensorflow_datasets as tfds
import transformers


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
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        # Initialize tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

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
        pt_sentences = []
        en_sentences = []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        # Load the pre-trained tokenizers
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True,
            clean_up_tokenization_spaces=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True,
            clean_up_tokenization_spaces=True)

        # Train both tokenizers on the dataset sentence iterators
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_sentences,
                                                            vocab_size=2 ** 13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_sentences,
                                                            vocab_size=2 ** 13)

        # Update the Dataset tokenizers with the newly trained ones
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

        return self.tokenizer_pt, self.tokenizer_en
