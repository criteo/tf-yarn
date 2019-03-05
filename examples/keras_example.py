"""
    Full example of using TF-Yarn to run Keras on YARN.
"""
import sys
import subprocess
import getpass
import gzip
import logging
import subprocess
import os

import numpy as np
import tensorflow as tf

from functools import partial
from datetime import datetime
from tensorflow import keras

from tf_yarn import event, TaskSpec, Experiment, run_on_yarn, packaging
import winequality

logging.basicConfig(level="INFO")


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
        train_data, test_data = winequality.get_train_eval_datasets(WINE_EQUALITY_FILE)

        def convert_to_tensor(x, y):
            return (tf.convert_to_tensor(list(x.values()), dtype=tf.float32),
                    tf.convert_to_tensor(y, dtype=tf.int32))

        def train_input_fn():
            return (train_data.map(convert_to_tensor)
                    .shuffle(1000)
                    .batch(128)
                    .repeat()
                    .make_one_shot_iterator()
                    .get_next())

        def eval_input_fn():
            return (test_data.map(convert_to_tensor)
                    .shuffle(1000)
                    .batch(128)
                    .make_one_shot_iterator()
                    .get_next())

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

    pyenv_zip_path, env_name = packaging.upload_env_to_hdfs()
    editable_requirements = packaging.get_editable_requirements_from_current_venv()
    run_on_yarn(
        pyenv_zip_path=pyenv_zip_path,
        experiment_fn=experiment_fn,
        task_specs={
            "chief": TaskSpec(memory=2 * 2 ** 10, vcores=4),
            "worker": TaskSpec(memory=2 * 2 ** 10, vcores=4, instances=5),
            "ps": TaskSpec(memory=2 * 2 ** 10, vcores=4, instances=2),
            "evaluator": TaskSpec(memory=2 ** 10, vcores=1)
        },
        files={
            **editable_requirements,
            os.path.basename(winequality.__file__): winequality.__file__,
        }
    )


if __name__ == "__main__":
    main()
