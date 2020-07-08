"""
    Full example of using TF-Yarn to run Keras on YARN.
"""
import logging
logging.basicConfig(level="INFO") # noqa
import getpass
import os

from datetime import datetime

import cluster_pack
from tf_yarn import TaskSpec, KerasExperiment, run_on_yarn
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


def main():
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
        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10),
            hvd.keras.callbacks.BroadcastGlobalVariablesCallback(0),
        ]
        train_params = {
            "steps_per_epoch": 1000,
            "callbacks": my_callbacks
        }
        return KerasExperiment(
            model=model,
            model_dir=HDFS_DIR,
            train_params=train_params,
            input_data_fn=input_data_fn,
            target_data_fn=None,
            validation_data_fn=validation_data_fn,
            session_config=None)

    pyenv_zip_path, _ = cluster_pack.upload_env()
    editable_requirements = cluster_pack.get_editable_requirements()
    run_on_yarn(
        pyenv_zip_path,
        experiment_fn,
        task_specs={
            "chief": TaskSpec(memory="2 GiB", vcores=4),
            "worker": TaskSpec(memory="2 GiB", vcores=4, instances=(HVD_SIZE - 1)),
            "evaluator": TaskSpec(memory="2 GiB", vcores=1)
        },
        files={
            **editable_requirements,
            os.path.basename(winequality.__file__): winequality.__file__,
        },
        custom_task_module="tf_yarn.tasks.gloo_allred_task"
    )


if __name__ == "__main__":
    main()
