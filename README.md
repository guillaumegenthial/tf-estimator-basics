# Tensorflow Estimator Basics

Train, predict, export and reload a `tf.estimator` for inference on a dummy example. 

[Read the blog post](https://guillaumegenthial.github.io/serving-tensorflow-estimator.html)




## Quickstart

```
make run
```

## Details

- `model.py` defines the `model_fn`
- `train.py` trains an Estimator using the `model_fn`
- `export.py` exports the Estimator as a `saved_model`
- `predict.py` reloads an Estimator and uses it for prediction
- `serve.py` reloads the inference graph from the `saved_model` format and uses it for prediction
