#!/usr/bin/env python3
"""module documentation
this module creates a function.

"""
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
    loss = loss of the network
    alpha = learning rate
    beta1 = momentum weight
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
    return optimizer
