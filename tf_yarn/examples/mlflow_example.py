"""
Example of using simple identity Estimator which just returns the input

1. Download winequality-red.csv from the Wine Quality dataset at UCI
   ML repository
   (https://archive.ics.uci.edu/ml/datasets/Wine+Quality).
2. Upload it to HDFS
"""


import logging
logging.basicConfig(level="INFO") # noqa
import os
import getpass
import requests
from datetime import datetime

import mlflow
import cluster_pack
from cluster_pack import filesystem
import winequality

from tf_yarn.tensorflow import Experiment, TaskSpec, run_on_yarn


logger = logging.getLogger(__name__)

USER = getpass.getuser()
WINE_EQUALITY_FILE = f"{cluster_pack.get_default_fs()}/user/{USER}/tf_yarn_test/winequality-red.csv"
# Output path of the learned model on hdfs
HDFS_DIR = (f"{cluster_pack.get_default_fs()}/user/{USER}"
            f"/tf_yarn_test/tf_yarn_{int(datetime.now().timestamp())}")


def _get_fs_for_tests():
    criteo_env = os.getenv("CRITEO_ENV", "")
    if criteo_env:
        return f"viewfs://{criteo_env}-am6"
    else:
        return cluster_pack.get_default_fs()


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
        model_dir=f"{HDFS_DIR}",
        n_classes=winequality.get_n_classes(),
        config=tf.estimator.RunConfig(
            save_checkpoints_steps=1000,
        ))
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

    # you need to install mlflow `pip install mlflow`
    # and set MLflow tracking uri
    mlflow.set_tracking_uri(os.getenv("CRITEO_MLFLOW_TRACKING_URI", ""))

    experiment_name = "tf-yarn-tests"
    exp = mlflow.get_experiment_by_name(experiment_name)
    if not exp:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            f"{_get_fs_for_tests()}/user/{USER}/mlflow_artifacts")
    else:
        experiment_id = exp.experiment_id

    run_id = mlflow.start_run(experiment_id=experiment_id).info.run_id

    run_on_yarn(
        experiment_fn,
        task_specs={
            "chief": TaskSpec(memory="1 GiB", vcores=1),
            "evaluator": TaskSpec(memory="1 GiB", vcores=1)
        },
        files={
            os.path.basename(winequality.__file__): winequality.__file__,
        }
    )

    mlflow.end_run()

    # check if run has been registered in MLFlow
    run_json = requests.get(f"{mlflow.get_tracking_uri()}/api/2.0/mlflow/runs/get",
                            params={'run_id': run_id}).json()

    logger.info(f"created run: {run_json}")

    metrics = run_json["run"]["data"]["metrics"]
    assert len(metrics) > 0
