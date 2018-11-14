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
import json
import os
import re
import sys
import traceback
import typing

import skein
import tensorflow as tf
import logging
import logging.config

from . import ExperimentFn
from ._internal import (
    iter_tasks,
    expand_wildcards_in_classpath,
    load_fn,
    xset_environ,
    reserve_sock_addr,
    MonitoredThread
)


def main(
    experiment_fn: ExperimentFn,
    all_tasks: typing.List[str],
    num_threads : int = 0
) -> None:


    def logs_event(logs: str) -> None:
        broadcast(client, f"{task}/logs", logs)

    def init_event(sock_addr: str) -> None:
        broadcast(client, f"{task}/init", sock_addr)

    def start_event() -> None:
        broadcast(client, f"{task}/start")

    def stop_event(e: typing.Optional[Exception]) -> None:
        broadcast(client, f"{task}/stop", maybe_format_exception(e))

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
    task = os.environ["SKEIN_CONTAINER_ID"].replace("_", ":", 1)
    task_type, task_str = task.split(":", 1)
    task_id: int = int(task_str)
    if task_type == "ps" and tf.test.is_gpu_available():
        tf.logging.warn(
            f"{task} is running on a GPU-enabled node. Consider setting "
            "label=NodeLabel.CPU in the corresponding TaskSpec.")

    client = skein.ApplicationClient.from_current()

    container = next(c for c in client.get_containers()
                     if c.yarn_container_id == os.environ["CONTAINER_ID"])
    logs = container.yarn_container_logs
    if logs is not None and not logs.startswith("http://"):
        logs = "http://" + logs
    logs_event(logs)

    # There is a race condition between acquiring a TCP port for
    # ``tf.train.Server``, and calling ``train_and_evaluate``.
    # There is no TensorFlow API to get rid of the race condition
    # completely, but the window of opportunity can be reduced by
    # preempting the server.
    # See https://github.com/tensorflow/tensorflow/issues/21492
    with reserve_sock_addr() as (host, port):
        init_event(f"{host}:{port}")
        spec = aggregate_spec(client, all_tasks)

        # Note that "evaluator" does not need a cluster, and "ps" (!)
        # surprisingly does not follow the same code path as the rest
        # and spawns a server regardless of the "environment" value.
        fake_google_env = task_type != "evaluator" and task_type != "ps"
        xset_environ(TF_CONFIG=json.dumps({
            "cluster": spec,
            "environment": "google" if fake_google_env else "",
            "task": {"type": task_type, "index": task_id},
        }))

        try:
            experiment = experiment_fn()
        except Exception as e:
            stop_event(e)
            raise

        config = experiment.config
        assert config.task_type == task_type and config.task_id == task_id
    
    server_config = tf.ConfigProto()
    server_config.intra_op_parallelism_threads=num_threads
    server_config.inter_op_parallelism_threads=num_threads
    if fake_google_env:

        tf.train.Server(
            config.cluster_spec,
            job_name=config.task_type,
            task_index=config.task_id,
            config=server_config,
            start=True)

    tf.logging.info(f"Starting {task_type}:{task_id}")
    thread = MonitoredThread(
        name=f"{task_type}:{task_id}",
        target=tf.estimator.train_and_evaluate,
        args=tuple(experiment),
        daemon=True)
    thread.start()
    start_event()

    # "ps" tasks do not terminate by themselves. See
    # https://github.com/tensorflow/tensorflow/issues/4713.
    if task_type != "ps":
        thread.join()
        tf.logging.info(f"{task_type}:{task_id} {thread.state}")

    # Wait for all tasks connected to this one. The set of tasks to
    # wait for contains all tasks in the cluster, or the ones
    # matching ``device_filters`` if set. The implementation assumes
    # that ``device_filers`` are symmetric.
    stop_event(thread.exception)
    wait_for_connected_tasks(
        client,
        all_tasks,
        getattr(config.session_config, "device_filters", []))

    if thread.exception is not None:
        raise thread.exception from None


def aggregate_spec(client, all_tasks):
    spec = {}
    for task in all_tasks:
        sock_addr = wait(client, f"{task}/init")
        task_type, _task_id = task.split(":", 1)
        spec.setdefault(task_type, []).append(sock_addr)
    return spec


def wait_for_connected_tasks(client, all_tasks, device_filters):
    for task in all_tasks:
        if matches_device_filters(task, device_filters):
            wait(client, f"{task}/stop")


def maybe_format_exception(e: typing.Optional[Exception]) -> str:
    if e is None:
        return ""
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))


def broadcast(
    client: skein.ApplicationClient,
    key: str,
    value: str = ""
) -> None:
    tf.logging.info(f"Broadcasting {key} = {value!r}")
    client.kv[key] = value.encode()


def wait(client: skein.ApplicationClient, key: str) -> str:
    tf.logging.info("Waiting for " + key)
    return client.kv.wait(key).decode()


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
    parser.add_argument("--experiment-fn", type=load_fn)
    parser.add_argument("--log-conf-file", type=str)
    parser.add_argument("--num-threads",type=int)
    args = parser.parse_args()
    _setup_logging(args.log_conf_file)
    main(args.experiment_fn, list(iter_tasks(args.num_workers, args.num_ps)),args.num_threads) 
