#!/usr/bin/env python3
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    optimizer = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
    return optimizer
