import logging
import os

import tensorflow as tf

import winequality

from tf_skein import Experiment, YARNCluster, TaskSpec, Env


def experiment_fn() -> Experiment:
    train, test = winequality.get_dataset()

    def train_input_fn():
        return (train.shuffle(1000)
                .batch(128)
                .repeat()
                .make_one_shot_iterator()
                .get_next())

    def eval_input_fn():
        return (test.shuffle(1000)
                .batch(128)
                .make_one_shot_iterator()
                .get_next())

    feature_columns = [
        tf.feature_column.numeric_column(name)
        for name in winequality.FEATURES
    ]

    config = tf.estimator.RunConfig(
        tf_random_seed=42,
        model_dir=f"hdfs://root/user/{os.environ['USER']}/dnn_classification")
    estimator = tf.estimator.DNNClassifier(
        hidden_units=[20, 20],
        feature_columns=feature_columns, n_classes=10,
        config=config)
    return Experiment(
        estimator,
        tf.estimator.TrainSpec(train_input_fn, max_steps=1000),
        tf.estimator.EvalSpec(
            eval_input_fn,
            start_delay_secs=0,
            throttle_secs=30))


if __name__ == "__main__":
    winequality.ensure_dataset_on_hdfs()

    cluster = YARNCluster(files={
        os.path.basename(winequality.__file__): winequality.__file__
    })
    cluster.run(experiment_fn, task_specs={
        "chief": TaskSpec(memory=2 * 2**10, vcores=4),
        "evaluator": TaskSpec(memory=2**10, vcores=1)
    })
