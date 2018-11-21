import logging
import os
import pwd
import sys
import time
from datetime import datetime
from functools import partial
from subprocess import check_output

import tensorflow as tf

import winequality
from tf_yarn import Experiment, run_on_yarn, TaskSpec

run_id = int(time.mktime(datetime.now().timetuple()))

def experiment_fn(dataset_path: str) -> Experiment:
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

    fs = check_output(
        "hdfs getconf -confKey fs.defaultFS".split()).strip().decode()
    user = pwd.getpwuid(os.getuid()).pw_name
    config = tf.estimator.RunConfig(
        tf_random_seed=42,
        model_dir=f"{fs}/user/{user}/examples/{run_id}")
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


if __name__ == "__main__":
    try:
        [dataset_path] = sys.argv[1:]
    except ValueError:
        sys.exit(winequality.__doc__)

    logging.basicConfig(level="INFO")
    
    #also one can use run_on_yarn(...,num_cores=num_cores) where num_cores is the user specified value, otherwise the default is 1
    run_on_yarn(
        partial(experiment_fn, dataset_path),
        task_specs={
            "chief": TaskSpec(memory=2 * 2 ** 10, vcores=4),
            "evaluator": TaskSpec(memory=2 ** 10, vcores=1)
        },
        files={os.path.basename(winequality.__file__): winequality.__file__}
        )