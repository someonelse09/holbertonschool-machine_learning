#!/usr/bin/env python3
"""This module includes the
function create_RMSProp_op sets up the
RMSProp optimization algorithm in TensorFlow"""

import tensorflow.keras as K


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Args:
        alpha is the learning rate
        beta2 is the RMSProp weight (Discounting factor)
        epsilon is a small number to avoid division by zero
    Returns:
         optimizer
    """
    return K.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
