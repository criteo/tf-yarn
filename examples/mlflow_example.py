"""
Example of using simple identity Estimator which just returns the input
"""
import logging
import os
import getpass
import mlflow
import skein
import tensorflow as tf

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
        n_classes=winequality.get_n_classes(),
        config=tf.estimator.RunConfig(
            save_checkpoints_steps=1000,
        ))
    return Experiment(
        estimator,
        tf.estimator.TrainSpec(train_input_fn, max_steps=10000),
        tf.estimator.EvalSpec(
            eval_input_fn,
            steps=100,
            start_delay_secs=0,
            throttle_secs=30))


if __name__ == "__main__":

    # you need to install mlflow `pip install mlflow`
    # and set MLFLOW_TRACKING_URI env variable
    # os.environ["MLFLOW_TRACKING_URI"] = "https://my-mlflow-server"
    mlflow.start_run()

    pyenv_zip_path, env_name = packaging.upload_env_to_hdfs()
    editable_requirements = packaging.get_editable_requirements_from_current_venv()

    run_on_yarn(
        pyenv_zip_path,
        experiment_fn,
        task_specs={
            "chief": TaskSpec(memory="1 GiB", vcores=1),
            "evaluator": TaskSpec(memory="1 GiB", vcores=1)
        },
        files={
            **editable_requirements,
            os.path.basename(winequality.__file__): winequality.__file__,
        }
    )

    mlflow.end_run()
