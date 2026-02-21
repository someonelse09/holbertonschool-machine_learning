#!/usr/bin/env python3
""" This module includes the function answer_loop
 that answers questions from a reference text"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """Args:
        reference is the reference text
    If the answer cannot be found in the reference text,
     respond with Sorry, I do not understand your question.
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

        answer = question_answer(user_input, reference)
        if answer is None or not answer.strip():
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))
