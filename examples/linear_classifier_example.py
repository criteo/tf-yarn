"""
Example of using LinearClassifier
"""
import logging
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

from tf_yarn import Experiment, TaskSpec, packaging, run_on_yarn
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


def experiment_fn() -> Experiment:
    train_data, test_data = winequality.get_train_eval_datasets(WINE_EQUALITY_FILE)

    def train_input_fn():
        return (train_data.shuffle(1000)
                .batch(128)
                .repeat()
                .make_one_shot_iterator()
                .get_next())

    def eval_input_fn():
        return (test_data.shuffle(1000)
                .batch(128)
                .make_one_shot_iterator()
                .get_next())

    estimator = tf.estimator.LinearClassifier(
        feature_columns=winequality.get_feature_columns(),
        model_dir=f"{HDFS_DIR}",
        n_classes=winequality.get_n_classes())
    return Experiment(
        estimator,
        tf.estimator.TrainSpec(train_input_fn, max_steps=10),
        tf.estimator.EvalSpec(
            eval_input_fn,
            steps=10,
            start_delay_secs=0,
            throttle_secs=30))


if __name__ == "__main__":
    pyenv_zip_path, env_name = packaging.upload_env_to_hdfs()
    editable_requirements = packaging.get_editable_requirements_from_current_venv()
    run_on_yarn(
        pyenv_zip_path,
        experiment_fn,
        task_specs={
            "chief": TaskSpec(memory=2 * 2 ** 10, vcores=4),
            "evaluator": TaskSpec(memory=2 ** 10, vcores=1)
        },
        files={
            **editable_requirements,
            os.path.basename(winequality.__file__): winequality.__file__,
        },
        acls=skein.model.ACLs(
            enable=True,
            view_users=['*']
        )
    )
