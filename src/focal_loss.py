#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Focal loss implementation
Code was implemented in https://github.com/mauriceqch/pcc_geo_cnn
"""

import tensorflow as tf
import tensorflow.keras.backend as K


def focal_loss(y_true, y_pred, gamma=2, alpha=0.6):
    """
    Computes the focal loss on geometry for training.
    
    Parameters:
    y_true: Original occupancy map.
    y_pred: Predicted probablities of occupancy.
     gamma: Factor to force the model to learn better harder samples.
     alpha: Factor that counteracts the class imbalance
            (1 occupied vs 0 unoccupied).
                      
    Output: The focal loss.
    """
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
