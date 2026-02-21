#!/usr/bin/env python3
"""This module includes the function question_answer
 that answers questions from multiple reference texts"""
qa_extract = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """
    Args:
        corpus_path is the path to the corpus of reference documents
    """
    exit_keywords = ["exit", "quit", "goodbye", "bye"]
    while True:
        try:
            user_input = input("Q: ")
        except EOFError:
            print("A: Goodbye")
            break
        if user_input.lower().strip() in exit_keywords:
            print("A: Goodbye")
            break
        print("A: ")

        reference = semantic_search(corpus_path, user_input)
        answer = qa_extract(user_input, reference)

        if answer is None or not answer.strip():
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))
