"""
    Full example of using TF-Yarn to run Keras on YARN.
"""
import logging
logging.basicConfig(level="INFO") # noqa
import getpass
import os

from datetime import datetime

import cluster_pack
from tf_yarn.tensorflow import TaskSpec, KerasExperiment, run_on_yarn, get_safe_experiment_fn
import winequality
import tensorflow as tf

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
WINE_EQUALITY_FILE = f"{cluster_pack.get_default_fs()}/user/{USER}/tf_yarn_test/winequality-red.csv"

"""
Output path of the learned model on hdfs
"""
HDFS_DIR = (f"{cluster_pack.get_default_fs()}/user/{USER}"
            f"/tf_yarn_test/tf_yarn_{int(datetime.now().timestamp())}")

HVD_SIZE = 2


def experiment_fn(hdfs_dir: str) -> KerasExperiment:
    def convert_to_tensor(x, y):
        return (tf.convert_to_tensor(value=list(x.values()), dtype=tf.float32),
                tf.convert_to_tensor(value=y, dtype=tf.int32))

    def input_data_fn():
        dataset = winequality.get_dataset(WINE_EQUALITY_FILE, split="train")
        return (dataset.map(convert_to_tensor)
                .shuffle(1000)
                .batch(128)
                .repeat())

    def validation_data_fn():
        dataset = winequality.get_dataset(WINE_EQUALITY_FILE, split="test")
        return (dataset.map(convert_to_tensor)
                .shuffle(1000)
                .batch(128))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=300, activation="relu", input_shape=(11,)))
    model.add(tf.keras.layers.Dense(units=100, activation="relu"))
    model.add(tf.keras.layers.Dense(units=10, activation="softmax"))
    model.summary()
    opt = tf.keras.optimizers.Adadelta(1.0 * HVD_SIZE)
    opt = hvd.DistributedOptimizer(opt)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    path_to_checkpoint = f"{hdfs_dir}" + "/checkpoint-{epoch}"
    my_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(path_to_checkpoint),
        hvd.keras.callbacks.BroadcastGlobalVariablesCallback(0),
    ]
    train_params = {
        "steps_per_epoch": 100,
        "callbacks": my_callbacks
    }
    return KerasExperiment(
        model=model,
        model_dir=hdfs_dir,
        train_params=train_params,
        input_data_fn=input_data_fn,
        target_data_fn=None,
        validation_data_fn=validation_data_fn)


# there seem to be pickling issues with Keras
# the experiment function is uploaded without pickling the experiment
# also see https://github.com/tensorflow/tensorflow/issues/32159
def get_safe_exp_fn():
    return get_safe_experiment_fn("native_keras_with_gloo_example.experiment_fn", HDFS_DIR)


def main():
    pyenv_zip_path, _ = cluster_pack.upload_env()
    editable_requirements = cluster_pack.get_editable_requirements()
    run_on_yarn(
        pyenv_zip_path,
        get_safe_exp_fn(),
        task_specs={
            "chief": TaskSpec(memory="2 GiB", vcores=4),
            "worker": TaskSpec(memory="2 GiB", vcores=4, instances=(HVD_SIZE - 1)),
            "evaluator": TaskSpec(memory="2 GiB", vcores=1)
        },
        files={
            **editable_requirements,
            os.path.basename(winequality.__file__): winequality.__file__,
            os.path.basename(__file__): __file__,
        },
        custom_task_module="tf_yarn.tensorflow.tasks.gloo_allred_task"
    )


if __name__ == "__main__":
    main()
