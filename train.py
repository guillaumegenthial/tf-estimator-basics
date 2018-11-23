"""Exporting a tf.estimator for prediction"""

__author__ = "Guillaume Genthial"

from pathlib import Path
import logging
import sys

import tensorflow as tf

from model import model_fn


def train_generator_fn():
    for number in range(100):
        yield [number, number], [2 * number]


def train_input_fn():
    shapes, types = (2, 1), (tf.float32, tf.float32)
    dataset = tf.data.Dataset.from_generator(
        train_generator_fn, output_types=types, output_shapes=shapes)
    dataset = dataset.batch(20).repeat(200)
    return dataset


if __name__ == '__main__':
    # Logging
    Path('model').mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler('model/train.log'),
        logging.StreamHandler(sys.stdout)
    ]
    logging.getLogger('tensorflow').handlers = handlers

    # Train estimator
    estimator = tf.estimator.Estimator(model_fn, 'model', params={})
    estimator.train(train_input_fn)
