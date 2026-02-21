#!/usr/bin/env python3
"""
This module contains the function prompt_loop
that takes input from the user and responds.
"""


def prompt_loop():
    """
    Continuous loop that prompts the user for a question.
    Exits on specific keywords: exit, quit, goodbye, bye.
    """
    exit_keywords = ["exit", "quit", "goodbye", "bye"]

    while True:
        try:
            input_word = input("Q: ").strip()
        except EOFError:
            break
        if input_word.lower() in exit_keywords:
            print("A: Goodbye")
            break
        # Print empty A: for non-exit inputs as per example
        print("A: ")

if __name__ == "__main__":
    prompt_loop()
