import tensorflow as tf
from typing import NamedTuple, Callable, Optional


class KerasExperiment(NamedTuple):
    model: tf.keras.models.Model
    model_dir: str
    train_params: dict
    input_data_fn: Optional[Callable]
    target_data_fn: Optional[Callable]
    validation_data_fn: Callable
