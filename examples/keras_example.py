"""
    Full example of using TF-Yarn to run Keras on YARN.
"""
import logging
logging.basicConfig(level="INFO") # noqa
import sys
import getpass
import gzip
import subprocess
import os
import skein
import numpy as np
import tensorflow as tf

from functools import partial
from datetime import datetime
from tensorflow import keras

from tf_yarn import event, TaskSpec, Experiment, run_on_yarn, packaging
import winequality

USER = getpass.getuser()

"""
1. Download winequality-*.csv from the Wine Quality dataset at UCI
   ML repository
   (https://archive.ics.uci.edu/ml/datasets/Wine+Quality).
2. Upload it to HDFS
3. Pass a full URI to either of the CSV files to the example
"""
WINE_EQUALITY_FILE = f"{packaging.get_default_fs()}/user/{USER}/tf_yarn_test/winequality-red.csv"

"""
Output path of the learned model on hdfs
"""
HDFS_DIR = (f"{packaging.get_default_fs()}/user/{USER}"
            f"/tf_yarn_test/tf_yarn_{int(datetime.now().timestamp())}")


def main():
    def experiment_fn() -> Experiment:

        def convert_to_tensor(x, y):
            return (tf.convert_to_tensor(value=list(x.values()), dtype=tf.float32),
                    tf.convert_to_tensor(value=y, dtype=tf.int32))

        def train_input_fn():
            dataset = winequality.get_dataset(WINE_EQUALITY_FILE, split="train")
            return (dataset.map(convert_to_tensor)
                    .shuffle(1000)
                    .batch(128)
                    .repeat())

        def eval_input_fn():
            dataset = winequality.get_dataset(WINE_EQUALITY_FILE, split="test")
            return (dataset.map(convert_to_tensor)
                    .shuffle(1000)
                    .batch(128))

        model = keras.Sequential()
        model.add(keras.layers.Dense(units=300, activation="relu", input_shape=(11,)))
        model.add(keras.layers.Dense(units=100, activation="relu"))
        model.add(keras.layers.Dense(units=10, activation="softmax"))
        model.summary()
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer="sgd",
                      metrics=['accuracy'])

        config = tf.estimator.RunConfig(model_dir=HDFS_DIR)
        estimator = tf.keras.estimator.model_to_estimator(model, config=config)
        return Experiment(
            estimator,
            tf.estimator.TrainSpec(
                train_input_fn,
                max_steps=1000),
            tf.estimator.EvalSpec(
                eval_input_fn,
                steps=10,
                start_delay_secs=0,
                throttle_secs=30))

    # forcing call to model_to_estimator._save_first_checkpoint l457
    # https://github.com/tensorflow/estimator/blob/ \
    # 1d55f01d8af871a35ef83fc3354b9feaa671cbe1/tensorflow_estimator/python/estimator/keras.py
    # otherwise there is a race condition
    # when all workers try to save the first checkpoint at the same time
    experiment_fn()

    pyenv_zip_path, env_name = packaging.upload_env_to_hdfs()
    editable_requirements = packaging.get_editable_requirements_from_current_venv()
    run_on_yarn(
        pyenv_zip_path,
        experiment_fn,
        task_specs={
            "chief": TaskSpec(memory="2 GiB", vcores=4),
            "worker": TaskSpec(memory="2 GiB", vcores=4, instances=4),
            "ps": TaskSpec(memory="2 GiB", vcores=4, instances=2),
            "evaluator": TaskSpec(memory="2 GiB", vcores=1)
        },
        files={
            **editable_requirements,
            os.path.basename(winequality.__file__): winequality.__file__,
        }
    )


if __name__ == "__main__":
    main()
