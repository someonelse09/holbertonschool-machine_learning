#!/usr/bin/env python3
"""
Semantic search module using Universal Sentence Encoder.
"""
import os
import tensorflow as tf
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents.

    Args:
        corpus_path (str): Path to the corpus of reference documents.
        sentence (str): Sentence from which to perform semantic search.

    Returns:
        str: The reference text of the document most similar to sentence.
    """
    url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = hub.load(url)

    documents = []

    # Iterate through the directory and read all files
    for filename in os.listdir(corpus_path):
        # Create full path and ensure we only process files
        file_path = os.path.join(corpus_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())

    if not documents:
        return None

    texts = [sentence] + documents

    embeddings = model(texts)

    # Separate the sentence embedding from the document embeddings
    sentence_emb = tf.expand_dims(embeddings[0], axis=0)
    docs_emb = embeddings[1:]

    # Calculate similarity using the dot product (cosine similarity
    # works similarly since USE vectors are approximately normalized).
    # transpose_b=True allows matrix multiplication to align dimensions.
    similarities = tf.matmul(sentence_emb, docs_emb, transpose_b=True)

    # Find the index of the maximum similarity score
    closest_idx = tf.argmax(similarities[0]).numpy()

    return documents[closest_idx]

