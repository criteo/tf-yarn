"""
Example of using LinearClassifier
"""
import logging
logging.basicConfig(level="INFO") # noqa
import os
import pwd
import getpass
import sys
import warnings
import typing
import skein
import tensorflow as tf

from functools import partial
from subprocess import check_output
from datetime import datetime

from tf_yarn import Experiment, TaskSpec, packaging, run_on_yarn, standalone_client_mode
import winequality
from tensorflow import keras
import numpy as np

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


def train_input_fn():
    dataset = winequality.get_dataset(WINE_EQUALITY_FILE, split="train")
    return (dataset.shuffle(1000)
            .batch(128)
            .repeat()
            )


def eval_input_fn():
    dataset = winequality.get_dataset(WINE_EQUALITY_FILE, split="test")
    return (dataset.shuffle(1000)
            .batch(128)
            )


if __name__ == "__main__":
    pyenv_zip_path, env_name = packaging.upload_env_to_hdfs()
    editable_requirements = packaging.get_editable_requirements_from_current_venv()

    with standalone_client_mode(
        pyenv_zip_path,
        task_specs={
            "worker": TaskSpec(memory="2 GiB", vcores=4, instances=2)
        },
        files=editable_requirements
    ) as cluster_spec:

        distrib_config = tf.contrib.distribute.DistributeConfig(
            train_distribute=tf.contrib.distribute.CollectiveAllReduceStrategy(),
            eval_distribute=tf.contrib.distribute.CollectiveAllReduceStrategy(),
            remote_cluster=cluster_spec
        )
        run_config = tf.estimator.RunConfig(
            experimental_distribute=distrib_config
        )

        estimator = tf.estimator.LinearClassifier(
            feature_columns=winequality.get_feature_columns(),
            model_dir=f"{HDFS_DIR}",
            n_classes=winequality.get_n_classes(),
            optimizer='Adam',
            config=run_config)

        tf.estimator.train_and_evaluate(
            estimator,
            tf.estimator.TrainSpec(train_input_fn, max_steps=1000),
            tf.estimator.EvalSpec(
                eval_input_fn,
                steps=10,
                start_delay_secs=0,
                throttle_secs=30))
