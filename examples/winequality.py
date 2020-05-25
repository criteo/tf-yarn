import typing

import tensorflow as tf

FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol"
]
LABEL = "quality"


def get_dataset(
    path: str,
    train_fraction: float = 0.7,
    split: str = "train"
) -> tf.data.Dataset:
    def split_label(*row):
        return dict(zip(FEATURES, row)), row[-1]

    def in_training_set(*row):
        num_buckets = 1000
        key = tf.strings.join(list(map(tf.as_string, row)))
        bucket_id = tf.strings.to_hash_bucket_fast(key, num_buckets)
        return bucket_id < int(train_fraction * num_buckets)

    def in_test_set(*row):
        return ~in_training_set(*row)

    data = tf.data.experimental.CsvDataset(
        path,
        [tf.float32] * len(FEATURES) + [tf.int32],
        header=True,
        field_delim=";")

    if split == "train":
        return data.filter(in_training_set).map(split_label)
    elif split == "test":
        return data.filter(in_test_set).map(split_label)
    else:
        raise ValueError("Unknown option split, must be 'train' or 'test'")


def get_feature_columns():
    return [tf.feature_column.numeric_column(name) for name in FEATURES]


def get_n_classes():
    return 10
