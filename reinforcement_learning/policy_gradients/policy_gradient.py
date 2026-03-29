#!/usr/bin/env python3
"""Module for computing policy using softmax function."""

import numpy as np


def policy(matrix, weight):
    """Compute the policy with a weight of a matrix.

    Args:
        matrix: input state matrix
        weight: weight matrix

    Returns:
        Softmax probabilities of the dot product of matrix and weight
    """
    z = matrix.dot(weight)
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def policy_gradient(state, weight):
    """Compute the Monte-Carlo policy gradient based on a state and weight.

    Args:
        state: matrix representing the current observation of the environment
        weight: matrix of random weight

    Returns:
        action: the chosen action (int)
        gradient: the gradient of the policy with respect to the weight matrix
    """
    state = state.reshape(1, -1)
    probs = policy(state, weight)

    action = np.random.choice(probs.shape[1], p=probs[0])

    d_softmax = probs.copy()
    d_softmax[0, action] -= 1
    gradient = state.T.dot(-d_softmax)

    return action, gradient
