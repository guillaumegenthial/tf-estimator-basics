"""Dummy model_fn"""

__author__ = "Guillaume Genthial"


import tensorflow as tf


def model_fn(features, labels, mode, params):
    # pylint: disable=unused-argument
    """Dummy model_fn"""
    if isinstance(features, dict):  # For serving
        features = features['feature']

    predictions = tf.layers.dense(features, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        loss = tf.nn.l2_loss(predictions - labels)
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(learning_rate=0.5).minimize(
                loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)
        else:
            raise NotImplementedError()
