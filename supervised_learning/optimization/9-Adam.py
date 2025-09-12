#!/usr/bin/env python3
"""This module includes the function
update_variables_Adam which updates a variable
 in place using the Adam optimization algorithm"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Args:
        alpha is the learning rate
        beta1 is the weight used for the first moment
        beta2 is the weight used for the second moment
        epsilon is a small number to avoid division by zero
        var is a numpy.ndarray containing the variable to be updated
        grad is a numpy.ndarray containing the gradient of var
        v is the previous first moment of var
        s is the previous second moment of var
        t is the time step used for bias correction
    Returns:
        the updated variable, the new first moment,
         and the new second moment, respectively
    """
    # First moment (mean) estimate:
    v = beta1 * v + (1 - beta1) * grad
    # Second moment (variance) estimate:
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    # Bias correction:
    v_adjusted = v / (1 - beta1 ** t)
    s_adjusted = s / (1 - beta2 ** t)
    # Final weight update:
    var = var - alpha * v_adjusted / (np.sqrt(s_adjusted) + epsilon)

    return var, v, s
