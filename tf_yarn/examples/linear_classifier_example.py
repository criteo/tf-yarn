"""
Example of using LinearClassifier

1. Download winequality-red.csv from the Wine Quality dataset at UCI
   ML repository
   (https://archive.ics.uci.edu/ml/datasets/Wine+Quality).
2. Upload it to HDFS
"""


import logging
logging.basicConfig(level="INFO") # noqa
import os
import getpass

from datetime import datetime

import cluster_pack
from cluster_pack import filesystem
import winequality

from tf_yarn.tensorflow import Experiment, TaskSpec, run_on_yarn


USER = getpass.getuser()
WINE_EQUALITY_FILE = f"{cluster_pack.get_default_fs()}/user/{USER}/tf_yarn_test/winequality-red.csv"
# Output path of the learned model on hdfs
HDFS_DIR = (f"{cluster_pack.get_default_fs()}/user/{USER}"
            f"/tf_yarn_test/tf_yarn_{int(datetime.now().timestamp())}")


def experiment_fn() -> Experiment:
    # To mitigate issue https://github.com/tensorflow/tensorflow/issues/32159 for tf >= 1.15
    import tensorflow as tf

    def train_input_fn():
        dataset = winequality.get_dataset(WINE_EQUALITY_FILE, split="train")
        return (dataset.shuffle(1000)
                .batch(128)
                .repeat())

    def eval_input_fn():
        dataset = winequality.get_dataset(WINE_EQUALITY_FILE, split="test")
        return (dataset.shuffle(1000)
                .batch(128))

    estimator = tf.estimator.LinearClassifier(
        feature_columns=winequality.get_feature_columns(),
        model_dir=HDFS_DIR,
        n_classes=winequality.get_n_classes())
    return Experiment(
        estimator,
        tf.estimator.TrainSpec(train_input_fn, max_steps=100),
        tf.estimator.EvalSpec(
            eval_input_fn,
            steps=10,
            start_delay_secs=0,
            throttle_secs=30))


if __name__ == "__main__":
    fs, _ = filesystem.resolve_filesystem_and_path(WINE_EQUALITY_FILE)
    if not fs.exists(WINE_EQUALITY_FILE):
        raise Exception(f"{WINE_EQUALITY_FILE} not found")

    run_on_yarn(
        experiment_fn,
        task_specs={
            "chief": TaskSpec(memory="2 GiB", vcores=4),
            "evaluator": TaskSpec(memory="2 GiB", vcores=1),
            "tensorboard": TaskSpec(memory="2 GiB", vcores=1, tb_model_dir=HDFS_DIR)
        },
        files={
            os.path.basename(winequality.__file__): winequality.__file__,
        }
    )
