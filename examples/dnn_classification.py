import os
import sys
import typing

sys.path.append(os.path.dirname(__file__))

import tensorflow as tf

from tf_skein import Experiment, YARNCluster, TaskSpec


# https://archive.ics.uci.edu/ml/datasets/Wine+Quality
FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol"
]
LABEL = "quality"


def get_dataset_path():
    path = f"hdfs://root/user/{os.environ['USER']}/winequality-red.csv"
    if not tf.gfile.Exists(path):
        tf.gfile.Copy(
            os.path.join(os.path.dirname(__file__), "winequality-red.csv"),
            path)
    return path


def get_dataset(
    path: str = None,
    train_fraction: float = 0.7
) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset]:
    path = path or get_dataset_path()

    def split_label(*row):
        return dict(zip(FEATURES, row)), row[-1]

    def in_training_set(*row):
        num_buckets = 1000
        key = tf.string_join(list(map(tf.as_string, row)))
        bucket_id = tf.string_to_hash_bucket_fast(key, num_buckets)
        return bucket_id < int(train_fraction * num_buckets)

    def in_test_set(*row):
        return ~in_training_set(*row)

    data = tf.contrib.data.CsvDataset(
        path,
        [tf.float32] * len(FEATURES) + [tf.int32],
        header=True,
        field_delim=";")

    train = data.filter(in_training_set).map(split_label).cache()
    test = data.filter(in_test_set).map(split_label).cache()
    return train, test


def config_fn():
    return tf.estimator.RunConfig(
        tf_random_seed=42,
        model_dir=f"hdfs://root/user/{os.environ['USER']}/dnn_classification")


def experiment_fn(config: tf.estimator.RunConfig) -> Experiment:
    train, test = get_dataset()

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
        for name in FEATURES
    ]

    return Experiment(
        tf.estimator.DNNClassifier(
            hidden_units=[20, 20],
            feature_columns=feature_columns,
            n_classes=10,
            config=config),
        tf.estimator.TrainSpec(train_input_fn, max_steps=1000),
        tf.estimator.EvalSpec(
            eval_input_fn,
            start_delay_secs=0,
            throttle_secs=30))


if __name__ == "__main__":
    get_dataset_path()  # Ensure the dataset is on HDFS.

    cluster = YARNCluster(task_specs={
        "chief": TaskSpec(memory=4 * 2 ** 10, vcores=8),
        "evaluator": TaskSpec(memory=2 * 2 ** 10, vcores=4)
    })

    tf.logging.set_verbosity("INFO")
    cluster.run(config_fn, experiment_fn)
