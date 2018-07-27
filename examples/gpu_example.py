import logging
import os

from tf_skein import TaskFlavor, TaskSpec, YARNCluster

import dnn_classifier_experiment as experiment_fn
import winequality


if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    cluster = YARNCluster(files={
        os.path.basename(winequality.__file__): winequality.__file__,
        os.path.basename(experiment_fn.__file__): experiment_fn.__file__,
    })
    cluster.run(experiment_fn.get, queue="ml-gpu", task_specs={
        "chief": TaskSpec(memory=2 * 2**10, vcores=4, flavor=TaskFlavor.GPU),
        "evaluator": TaskSpec(memory=2**10, vcores=1)
    })
