#!/usr/bin/env python3
"""This module defines a function
which updates the learning rate
using inverse time decay in numpy"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Args:
        alpha is the original learning rate
        decay_rate is the weight used to determine
            the rate at which alpha will decay
        global_step is the number of passes of
            gradient descent that have elapsed
        decay_step is the number of passes of gradient descent
            that should occur before alpha is decayed further
        the learning rate decay should occur in a stepwise fashion
    Returns:
        the updated value for alpha
    """
    decay_factor = np.floor(global_step / decay_step)
    return alpha / (1 + decay_factor * decay_rate)
