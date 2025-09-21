#!/usr/bin/env python3
"""This module includes the function that
calculates the cost of a neural network with L2 regularization"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Args:
        cost is a tensor containing the cost of the network without L2 regularization
        model is a Keras model that includes layers with L2 regularization
    Returns:
        a tensor containing the total cost for each layer of the network, accounting for L2 regularization
    """
    # Get all regularization losses from the model
    regularization_losses = []

    for reg_loss in model.losses:
        regularization_losses.append(tf.reduce_sum(reg_loss))
    total_cost = cost + tf.stack(regularization_losses)
    return total_cost
