import os
import pwd
from subprocess import check_output

import tensorflow as tf

from tf_yarn import Experiment

import winequality


def get(dataset_path: str) -> Experiment:
    train_data, test_data = winequality.get_train_eval_datasets(dataset_path)

    def train_input_fn():
        return (train_data.shuffle(1000)
                .batch(128)
                .repeat()
                .make_one_shot_iterator()
                .get_next())

    def eval_input_fn():
        return (test_data.shuffle(1000)
                .batch(128)
                .make_one_shot_iterator()
                .get_next())

    # XXX the fs.defaultFS part is to make the examples work inside
    #     ``hadoop-test-cluster``.
    fs = check_output(
        "hdfs getconf -confKey fs.defaultFS".split()).strip().decode()
    user = pwd.getpwuid(os.getuid()).pw_name
    config = tf.estimator.RunConfig(
        tf_random_seed=42,
        model_dir=f"{fs}/user/{user}/{__name__}")
    estimator = tf.estimator.LinearClassifier(
        winequality.get_feature_columns(),
        n_classes=winequality.get_n_classes(),
        config=config)
    return Experiment(
        estimator,
        tf.estimator.TrainSpec(train_input_fn, max_steps=10),
        tf.estimator.EvalSpec(
            eval_input_fn,
            steps=10,
            start_delay_secs=0,
            throttle_secs=30))
