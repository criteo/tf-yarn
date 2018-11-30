# Copyright 2018 Criteo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
import sys
import typing

import dill
import skein
import tensorflow as tf
import logging
import logging.config

from . import ExperimentFn
from ._internal import (
    iter_tasks,
    expand_wildcards_in_classpath,
    MonitoredThread
)
from . import event
from . import cluster


def main(all_tasks: typing.List[str]) -> None:

    tf.logging.info("Python " + sys.version)
    tf.logging.info("Skein " + skein.__version__)
    tf.logging.info(f"TensorFlow {tf.GIT_VERSION} {tf.VERSION}")

    try:
        classpath = os.environ["CLASSPATH"]
    except KeyError:
        tf.logging.warn(
            "$CLASSPATH is not defined. HDFS access will surely fail.")
    else:
        tf.logging.info("Attempting to expand wildcards in $CLASSPATH")
        os.environ["CLASSPATH"] = expand_wildcards_in_classpath(classpath)
        del classpath
        tf.logging.info(os.environ["CLASSPATH"])

    # XXX this assumes no service restarts, because after a restart
    #     the task_id might exceed ``num_workers`` or ``num_ps``.
    task = cluster.get_task()
    task_type, task_id = cluster.get_task_description()
    if task_type == "ps" and tf.test.is_gpu_available():
        tf.logging.warn(
            f"{task} is running on a GPU-enabled node. Consider setting "
            "label=NodeLabel.CPU in the corresponding TaskSpec.")

    client = skein.ApplicationClient.from_current()

    event.broadcast_container_start_time(client, task)

    container = next(c for c in client.get_containers()
                     if c.yarn_container_id == os.environ["CONTAINER_ID"])
    logs = container.yarn_container_logs
    if logs is not None and not logs.startswith("http://"):
        logs = "http://" + logs
    event.logs_event(client, task, logs)

    cluster_spec = cluster.start_cluster(client, all_tasks)

    try:
        experiment = dill.loads(client.kv.wait('experiment_fn'))()
    except Exception as e:
        event.stop_event(client, task, e)
        raise

    config = experiment.config
    cluster.start_tf_server(cluster_spec, config.session_config)

    tf.logging.info(f"Starting {task_type}:{task_id}")
    thread = MonitoredThread(
        name=f"{task_type}:{task_id}",
        target=gen_monitored_train_and_evaluate(client, task),
        args=tuple(experiment),
        daemon=True)

    thread.start()
    event.start_event(client, task)

    # "ps" tasks do not terminate by themselves. See
    # https://github.com/tensorflow/tensorflow/issues/4713.
    if task_type != "ps":
        thread.join()
        tf.logging.info(f"{task_type}:{task_id} {thread.state}")

    # Wait for all tasks connected to this one. The set of tasks to
    # wait for contains all tasks in the cluster, or the ones
    # matching ``device_filters`` if set. The implementation assumes
    # that ``device_filers`` are symmetric.
    event.stop_event(client, task, thread.exception)
    wait_for_connected_tasks(
        client,
        all_tasks,
        getattr(config.session_config, "device_filters", []))

    event.broadcast_container_stop_time(client, task)

    if thread.exception is not None:
        raise thread.exception from None


def gen_monitored_train_and_evaluate(client: skein.ApplicationClient,
                                     task: str):
    def train_and_evaluate(
            estimator: tf.estimator,
            train_spec: tf.estimator.TrainSpec,
            eval_spec: tf.estimator.EvalSpec):
        event.broadcast_train_eval_start_timer(client, task)
        tf.estimator.train_and_evaluate(
            estimator,
            train_spec,
            eval_spec
        )
        event.broadcast_train_eval_stop_timer(client, task)
    return train_and_evaluate


def wait_for_connected_tasks(client, all_tasks, device_filters):
    for task in all_tasks:
        if matches_device_filters(task, device_filters):
            event.wait(client, f"{task}/stop")


def matches_device_filters(task: str, device_filters: typing.List[str]):
    task_type, task_id = task.split(":", 1)
    for device_filter in device_filters:
        [(filter_type, filter_id)] = re.findall(
            r"^/job:([a-z]+)(?:/task:(\d+))?$",
            # Remove once https://github.com/tensorflow/tensorflow/pull/22566 is released
            device_filter.replace("master", "chief"))
        if (filter_type == task_type and
                (not filter_id or filter_id == task_id)):
            return True
    return not device_filters


def _setup_logging(log_conf_file):
    if log_conf_file is None:
        base_dir = os.path.dirname(sys.modules["tf_yarn"].__file__)
        log_conf_file = os.path.join(base_dir, "default.log.conf")

    logging.config.fileConfig(log_conf_file, disable_existing_loggers=True)
    tf.logging.info(f"using log file {log_conf_file}")


if __name__ == "__main__":
    print(f"start", file=sys.stderr)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--num-ps", type=int)
    parser.add_argument("--log-conf-file", type=str)
    args = parser.parse_args()
    _setup_logging(args.log_conf_file)
    main(list(iter_tasks(args.num_workers, args.num_ps)))
