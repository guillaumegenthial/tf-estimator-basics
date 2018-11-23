"""Export estimator as a saved_model"""

__author__ = "Guillaume Genthial"

import tensorflow as tf

from model import model_fn


def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    number = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='number')
    receiver_tensors = {'number': number}
    features = tf.tile(number, multiples=[1, 2])
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


if __name__ == '__main__':
    estimator = tf.estimator.Estimator(model_fn, 'model', params={})
    estimator.export_saved_model('saved_model', serving_input_receiver_fn)
