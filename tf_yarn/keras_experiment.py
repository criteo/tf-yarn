import tensorflow as tf
from typing import NamedTuple


class KerasExperiment(NamedTuple):
    model: tf.keras.models.Model
    model_dir: str
    session_config: tf.compat.v1.ConfigProto
