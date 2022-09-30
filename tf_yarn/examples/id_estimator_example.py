"""
Example of using simple identity Estimator which just returns the input
"""


import logging
logging.basicConfig(level="INFO") # noqa

import skein

from tf_yarn.tensorflow import Experiment, TaskSpec, run_on_yarn


def model_fn(features, labels, mode):
    # To mitigate issue https://github.com/tensorflow/tensorflow/issues/32159 for tf >= 1.15
    import tensorflow as tf

    x = features["x"]
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"x": x},
            export_outputs={})

    loss = tf.compat.v1.losses.mean_squared_error(x, labels)
    train_op = tf.compat.v1.assign_add(tf.compat.v1.train.get_global_step(), 1)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        predictions={"x": x},
        eval_metric_ops={})


def experiment_fn() -> Experiment:
    # To mitigate issue https://github.com/tensorflow/tensorflow/issues/32159 for tf >= 1.15
    import tensorflow as tf

    def input_fn():
        x = tf.constant([[1.0], [2.0], [3.0], [4.0]])
        return {"x": x}, x

    estimator = tf.estimator.Estimator(model_fn=model_fn)
    train_spec = tf.estimator.TrainSpec(input_fn, max_steps=1)
    eval_spec = tf.estimator.EvalSpec(input_fn, steps=1)
    return Experiment(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    # skein.Client is useful when multiple learnings run in parallel
    # and share one single skein JAVA process
    with skein.Client() as client:
        run_on_yarn(
            experiment_fn,
            task_specs={
                "chief": TaskSpec(memory="1 GiB", vcores=1)
            },
            skein_client=client
        )
