import argparse
import json
import os
import re
import sys
import typing

import skein
import tensorflow as tf

from . import ExperimentFn
from ._internal import (
    MonitoredThread,
    reserve_sock_addr,
    load_fn,
    xset_environ
)


def main(
    experiment_fn: ExperimentFn,
    all_tasks: typing.List[str]
) -> None:
    tf.logging.info("Python " + sys.version)
    tf.logging.info("Skein " + skein.__version__)
    tf.logging.info(f"TensorFlow {tf.GIT_VERSION} {tf.VERSION}")

    # XXX this assumes no service restarts, because after a restart
    #     the task_id might exceed ``num_workers`` or ``num_ps``.
    task = os.environ["SKEIN_CONTAINER_ID"].replace("_", ":", 1)
    task_type, task_id = task.split(":", 1)
    task_id = int(task_id)
    client = skein.ApplicationClient.from_current()

    # There is a race condition between acquiring a TCP port for
    # ``tf.train.Server``, and calling ``train_and_evaluate``.
    # There is no TensorFlow API to get rid of the race condition
    # completely, but the window of opportunity can be reduced by
    # preempting the server.
    # See https://github.com/tensorflow/tensorflow/issues/21492
    with reserve_sock_addr() as (host, port):
        broadcast(client, "init/" + task, f"{host}:{port}")
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
        experiment = experiment_fn()
        config = experiment.config
        assert config.task_type == task_type and config.task_id == task_id

    if fake_google_env:
        tf.train.Server(
            config.cluster_spec,
            job_name=config.task_type,
            task_index=config.task_id,
            config=config.session_config,
            start=True)

    tf.logging.info(f"Starting {task_type}:{task_id}")
    thread = MonitoredThread(
        name=f"{task_type}:{task_id}",
        target=tf.estimator.train_and_evaluate,
        args=tuple(experiment),
        # "ps" tasks do not terminate by themselves. See
        # https://github.com/tensorflow/tensorflow/issues/4713
        daemon=task_type == "ps")
    thread.start()

    if task_type != "ps":
        thread.join()
        tf.logging.info(f"{task_type}:{task_id} {thread.state}")

    # Wait for all tasks connected to this one. The set of tasks to
    # wait for contains all tasks in the cluster, or the ones
    # matching ``device_filters`` if set. The implementation assumes
    # that ``device_filers`` are symmetric.
    broadcast(client, "stop/" + task)
    wait_for_connected_tasks(
        client,
        all_tasks,
        getattr(config.session_config, "device_filters", []))

    if thread.exception is not None:
        raise thread.exception from None


def aggregate_spec(client, all_tasks):
    spec = {}
    for task in all_tasks:
        sock_addr = wait(client, "init/" + task)
        task_type, _task_id = task.split(":", 1)
        spec.setdefault(task_type, []).append(sock_addr)
    return spec


def wait_for_connected_tasks(client, all_tasks, device_filters):
    for task in all_tasks:
        if matches_device_filters(task, device_filters):
            wait(client, "stop/" + task)


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
            "^/job:([a-z]+)(?:/task:(\d+))?$",
            device_filter)
        if (filter_type == task_type and
                (not filter_id or filter_id == task_id)):
            return True
    return not device_filters


def iter_tasks(num_workers, num_ps):
    yield "chief:0"
    yield from (f"worker:{task_id}" for task_id in range(num_workers))
    yield from (f"ps:{task_id}" for task_id in range(num_ps))


if __name__ == "__main__":
    tf.logging.set_verbosity("INFO")

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--num-ps", type=int)
    parser.add_argument("--experiment-fn", type=load_fn)
    args = parser.parse_args()
    main(args.experiment_fn, list(iter_tasks(args.num_workers, args.num_ps)))
