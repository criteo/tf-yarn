import logging
import os

import dnn_classifier_experiment as experiment_fn
import winequality

from tf_yarn import run_on_yarn, TaskSpec

if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    winequality.ensure_dataset_on_hdfs()

    run_on_yarn(
        experiment_fn.get,
        task_specs={
            "chief": TaskSpec(memory=2 * 2 ** 10, vcores=4),
            "evaluator": TaskSpec(memory=2 ** 10, vcores=1)
        },
        files={
            os.path.basename(winequality.__file__): winequality.__file__,
            os.path.basename(experiment_fn.__file__): experiment_fn.__file__,
        }
    )
