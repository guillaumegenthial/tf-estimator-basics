"""Reload and serve a saved model"""

__author__ = "Guillaume Genthial"

from pathlib import Path
import time

from tensorflow.contrib import predictor


def my_service():
    """Some service yielding numbers"""
    start, end = 100, 110
    for number in range(start, end):
        yield number


if __name__ == '__main__':
    export_dir = 'saved_model'
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(latest)
    tic = time.time()
    for nb in my_service():
        pred = predict_fn({'number': [[nb]]})['output']
        # print((pred - 2*nb)**2)
    toc = time.time()
    print('Average time in serve.py: {}s'.format((toc - tic) / 10))
