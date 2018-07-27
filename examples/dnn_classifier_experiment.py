import os

import tensorflow as tf

from tf_skein import Experiment

import winequality
winequality.ensure_dataset_on_hdfs()


def get() -> Experiment:
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
