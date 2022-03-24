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

import cluster_pack
from cluster_pack import filesystem
from tf_yarn.tensorflow import TaskSpec, KerasExperiment, run_on_yarn
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
WINE_EQUALITY_FILE = f"{cluster_pack.get_default_fs()}/user/{USER}/tf_yarn_test/winequality-red.csv"

# Output path of the learned model on hdfs
HDFS_DIR = (f"{cluster_pack.get_default_fs()}/user/{USER}"
            f"/tf_yarn_test/tf_yarn_{int(datetime.now().timestamp())}")
HVD_SIZE = 2


def experiment_fn() -> KerasExperiment:
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
    path_to_checkpoint = f"{HDFS_DIR}" + "/checkpoint-{epoch}"
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
        model_dir=HDFS_DIR,
        train_params=train_params,
        input_data_fn=input_data_fn,
        target_data_fn=None,
        validation_data_fn=validation_data_fn)


def main():
    fs, _ = filesystem.resolve_filesystem_and_path(WINE_EQUALITY_FILE)
    if not fs.exists(WINE_EQUALITY_FILE):
        raise Exception(f"{WINE_EQUALITY_FILE} not found")

    run_on_yarn(
        experiment_fn,
        task_specs={
            "chief": TaskSpec(memory="2 GiB", vcores=4),
            "worker": TaskSpec(memory="2 GiB", vcores=4, instances=(HVD_SIZE - 1)),
            "evaluator": TaskSpec(memory="2 GiB", vcores=1)
        },
        files={
            os.path.basename(winequality.__file__): winequality.__file__,
            os.path.basename(__file__): __file__,
        },
        custom_task_module="tf_yarn.tensorflow.tasks.gloo_allred_task"
    )


if __name__ == "__main__":
    main()
