"""
Full example of using TF-Yarn to run Keras on YARN.

1. Download winequality-red.csv from the Wine Quality dataset at UCI
    ML repository
    (https://archive.ics.uci.edu/ml/datasets/Wine+Quality).
2. Upload it to HDFS
"""


import logging
logging.basicConfig(level="INFO") # noqa
import getpass
import os
from datetime import datetime

from tensorflow import keras
import cluster_pack
from cluster_pack import filesystem
import winequality
import tensorflow as tf

from tf_yarn.tensorflow import TaskSpec, Experiment, run_on_yarn


logger = logging.getLogger()
USER = getpass.getuser()
WINE_EQUALITY_FILE = f"{cluster_pack.get_default_fs()}/user/{USER}/tf_yarn_test/winequality-red.csv"
# Output path of the learned model on hdfs
HDFS_DIR = (f"{cluster_pack.get_default_fs()}/user/{USER}"
            f"/tf_yarn_test/tf_yarn_{int(datetime.now().timestamp())}")


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
            max_steps=100),
        tf.estimator.EvalSpec(
            eval_input_fn,
            steps=10,
            start_delay_secs=0,
            throttle_secs=30))


def main():
    fs, _ = filesystem.resolve_filesystem_and_path(WINE_EQUALITY_FILE)
    if not fs.exists(WINE_EQUALITY_FILE):
        raise Exception(f"{WINE_EQUALITY_FILE} not found")

    # forcing call to model_to_estimator._save_first_checkpoint l457
    # https://github.com/tensorflow/estimator/blob/ \
    # 1d55f01d8af871a35ef83fc3354b9feaa671cbe1/tensorflow_estimator/python/estimator/keras.py
    # otherwise there is a race condition
    # when all workers try to save the first checkpoint at the same time
    experiment_fn()

    run_on_yarn(
        experiment_fn,
        task_specs={
            "chief": TaskSpec(memory="2 GiB", vcores=4),
            "worker": TaskSpec(memory="2 GiB", vcores=4, instances=4),
            "ps": TaskSpec(memory="2 GiB", vcores=4, instances=2),
            "evaluator": TaskSpec(memory="2 GiB", vcores=1)
        },
        files={
            os.path.basename(winequality.__file__): winequality.__file__,
            os.path.basename(__file__): __file__
        }
    )


if __name__ == "__main__":
    main()
