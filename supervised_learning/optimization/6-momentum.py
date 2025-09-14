#!/usr/bin/env python3
"""This module includes the function that
sets up the gradient descent with momentum
optimization algorithm in TensorFlow"""

import tensorflow.keras as K


def create_momentum_op(alpha, beta1):
    """
    Args:
        alpha is the learning rate.
        beta1 is the momentum weight.
    Returns: optimizer
    """

    return K.optimizers.SGD(learning_rate=alpha, momentum=beta1)
