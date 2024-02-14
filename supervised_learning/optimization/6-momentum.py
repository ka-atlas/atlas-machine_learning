#!/usr/bin/env python3
import tensorflow.compat.v1 as tf

def create_momentum_op(loss, alpha, beta1):
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
