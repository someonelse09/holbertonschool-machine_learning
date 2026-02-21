#!/usr/bin/env python3
"""
This module includes the function question_answer that
finds a snippet of text within a reference document to answer a question.
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """

    Args:
        question is a string containing the question to answer
        reference is a string containing the reference
         document from which to find the answer
    Returns:
        a string containing the answer
        If no answer is found, return None
        Your function should use the bert-uncased-tf2-qa model
         from the tensorflow-hub library
        Your function should use the pre-trained BertTokenizer,
         bert-large-uncased-whole-word-masking-finetuned-squad,
          from the transformers library
    """
    # Load the pre-trained tokenizer
    tokenizer_nm = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_nm)

    # Load the pre-trained model from TensorFlow Hub
    model_url = "https://tfhub.dev/see--/bert-uncased-tf2-qa/1"
    model = hub.load(model_url)

    # Tokenize the question and reference text
    inputs = tokenizer(question, reference, return_tensors='tf')

    input_word_ids = inputs['input_ids']
    input_mask = inputs['attention_mask']
    input_type_ids = inputs['token_type_ids']

    # Run the model
    outputs = model([input_word_ids, input_mask, input_type_ids])

    # Get the start and end logits
    start_logits = outputs[0]
    end_logits = outputs[1]

    # Determine the most likely start and end indices
    start_index = tf.argmax(start_logits[0])
    end_index = tf.argmax(end_logits[0])

    # Check if the answer exists or is logical
    if start_index == 0 or end_index == 0 or start_index > end_index:
        return None

    # Convert token ids back to tokens and then to a string
    tokens = tokenizer.convert_ids_to_tokens(input_word_ids[0])
    answer_tokens = tokens[start_index:end_index + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    # Handle cases where BERT returns an empty string or just special tokens
    if not answer.strip():
        return None

    return answer
