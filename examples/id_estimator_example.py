"""
Example of using simple identity Estimator which just returns the input
"""
import logging
import os
import pwd
import getpass
from subprocess import check_output
import skein
import tensorflow as tf

from tf_yarn import Experiment, TaskSpec, packaging, run_on_yarn

logging.basicConfig(level="INFO")


def model_fn(features, labels, mode):
    x = features["x"]
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"x": x},
            export_outputs={})

    loss = tf.losses.mean_squared_error(x, labels)
    train_op = tf.assign_add(tf.train.get_global_step(), 1)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        predictions={"x": x},
        eval_metric_ops={})


def experiment_fn() -> Experiment:
    def input_fn():
        x = tf.constant([[1.0], [2.0], [3.0], [4.0]])
        return {"x": x}, x

    estimator = tf.estimator.Estimator(model_fn=model_fn)
    train_spec = tf.estimator.TrainSpec(input_fn, max_steps=1)
    eval_spec = tf.estimator.EvalSpec(input_fn, steps=1)
    return Experiment(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    pyenv_zip_path, env_name = packaging.upload_env_to_hdfs()
    editable_requirements = packaging.get_editable_requirements_from_current_venv()
    # skein.Client is useful when multiple learnings run in parallel
    # and share one single skein JAVA process
    with skein.Client() as client:
        run_on_yarn(
            pyenv_zip_path,
            experiment_fn,
            task_specs={
                "chief": TaskSpec(memory="1 GiB", vcores=1)
            },
            files={
                **editable_requirements,
            },
            skein_client=client
        )
