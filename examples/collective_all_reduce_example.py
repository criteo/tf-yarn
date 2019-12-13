"""
Example of using horovod
"""
import logging
logging.basicConfig(level="INFO") # noqa
import os
import getpass
import tensorflow as tf
import winequality
from datetime import datetime
from tf_yarn import Experiment, TaskSpec, packaging, run_on_yarn

logger = logging.getLogger(__name__)

try:
    import horovod.tensorflow as hvd
except (ModuleNotFoundError):
    logger.warning(
       "horovod not installed. checkout "
       "https://github.com/criteo/tf-yarn/blob/master/docs/HorovodWithGloo.md"
    )
    pass

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
    def train_input_fn():
        dataset = winequality.get_dataset(WINE_EQUALITY_FILE, split="train")
        return dataset.shuffle(1000).batch(128).repeat()

    def eval_input_fn():
        dataset = winequality.get_dataset(WINE_EQUALITY_FILE, split="test")
        return dataset.shuffle(1000).batch(128)

    estimator = tf.estimator.LinearClassifier(
        feature_columns=winequality.get_feature_columns(),
        model_dir=f"{HDFS_DIR}",
        n_classes=winequality.get_n_classes(),
        optimizer=lambda: hvd.DistributedOptimizer(tf.train.AdamOptimizer())
    )

    return Experiment(
        estimator,
        tf.estimator.TrainSpec(
            train_input_fn,
            max_steps=10,
            hooks=[hvd.BroadcastGlobalVariablesHook(0)]
        ),
        tf.estimator.EvalSpec(
            eval_input_fn,
            steps=10,
            start_delay_secs=0,
            throttle_secs=30
        )
    )


if __name__ == "__main__":
    pyenv_zip_path, _ = packaging.upload_env_to_hdfs()
    editable_requirements = packaging.get_editable_requirements_from_current_venv()

    run_on_yarn(
        pyenv_zip_path,
        experiment_fn,
        task_specs={
            "chief": TaskSpec(memory="2 GiB", vcores=4),
            "worker": TaskSpec(memory="2 GiB", vcores=4, instances=1),
            "evaluator": TaskSpec(memory="2 GiB", vcores=1),
            "tensorboard": TaskSpec(memory="2 GiB", vcores=1, tb_model_dir=HDFS_DIR)
        },
        files={
            **editable_requirements,
            os.path.basename(winequality.__file__): winequality.__file__,
        },
        custom_task_module="tf_yarn.tasks._gloo_allred_task"
    )
