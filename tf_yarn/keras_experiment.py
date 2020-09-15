import tensorflow as tf
from typing import NamedTuple, Callable


class KerasExperiment(NamedTuple):
    model: tf.keras.models.Model
    model_dir: str
    train_params: dict
    input_data_fn: Callable
    target_data_fn: Callable
    validation_data_fn: Callable
