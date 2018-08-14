"""Wine quality dataset from UCI ML repository.

https://archive.ics.uci.edu/ml/datasets/Wine+Quality
"""

import os
import pwd
import typing
from subprocess import check_output

import tensorflow as tf

FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol"
]
LABEL = "quality"


def ensure_dataset_on_hdfs():
    path = get_dataset_hdfs_path()
    if not tf.gfile.Exists(path):
        tf.gfile.Copy(
            os.path.join(os.path.dirname(__file__), "winequality-red.csv"),
            path)


def get_dataset_hdfs_path():
    # XXX the fs.defaultFS part is to make the examples work inside
    #     ``hadoop-test-cluster``.
    fs = check_output(
        "hdfs getconf -confKey fs.defaultFS".split()).strip().decode()
    user = pwd.getpwuid(os.getuid()).pw_name
    return fs + f"/user/{user}/winequality-red.csv"


def get_dataset(
    train_fraction: float = 0.7
) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset]:
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
        get_dataset_hdfs_path(),
        [tf.float32] * len(FEATURES) + [tf.int32],
        header=True,
        field_delim=";")

    train = data.filter(in_training_set).map(split_label).cache()
    test = data.filter(in_test_set).map(split_label).cache()
    return train, test
