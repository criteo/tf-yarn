import os
import pwd
from subprocess import check_output

import tensorflow as tf

from tf_skein import Experiment

import winequality


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

    # XXX the fs.defaultFS part is to make the examples work inside
    #     ``hadoop-test-cluster``.
    fs = check_output(
        "hdfs getconf -confKey fs.defaultFS".split()).strip().decode()
    user = pwd.getpwuid(os.getuid()).pw_name
    config = tf.estimator.RunConfig(
        tf_random_seed=42,
        model_dir=f"{fs}/user/{user}/dnn_classification")
    estimator = tf.estimator.DNNClassifier(
        hidden_units=[10, 10],
        feature_columns=feature_columns, n_classes=10,
        config=config)
    return Experiment(
        estimator,
        tf.estimator.TrainSpec(train_input_fn, max_steps=10),
        tf.estimator.EvalSpec(
            eval_input_fn,
            steps=10,
            start_delay_secs=0,
            throttle_secs=30))
