import argparse
import json
import logging
import os
from contextlib import closing
from functools import partial

import skein
import tensorflow as tf

from ._internal import (
    MonitoredThread,
    iter_available_sock_addrs,
    decode_fn,
    spec_from_kv,
    xset_environ
)
from .cluster import ExperimentFn


def main(
    experiment_fn: ExperimentFn,
    num_workers: int,
    num_ps: int
) -> None:
    def broadcast(key: str, value: str = ""):
        client.kv[key] = value.encode()

    task = os.environ["SKEIN_CONTAINER_ID"]
    task_type, task_id = task.split("_", 1)
    task_id = int(task_id)
    client = skein.ApplicationClient.from_current()

    with closing(iter_available_sock_addrs()) as it:
        host, port = next(it)
        # Broadcast the addr to other tasks, and block until the full
        # cluster spec is available.
        broadcast("init/" + task, f"{host}:{port}")
        spec = spec_from_kv(client.kv, "init", num_workers, num_ps)

        # XXX this code is within the ``with`` block to keep the socket
        #     reserved until everything is ready to ``train_and_evaluate``.
        xset_environ(TF_CONFIG=json.dumps({
            "cluster": spec,
            "task": {"type": task_type, "index": task_id}
        }))
        experiment = experiment_fn()
        config = experiment.config
        assert config.task_type == task_type and config.task_id == task_id

    thread = MonitoredThread(
        name=f"{task_type}:{task_id}",
        target=partial(tf.estimator.train_and_evaluate, *experiment),
        # "ps" tasks do not terminate by themselves. See
        # https://github.com/tensorflow/tensorflow/issues/4713
        daemon=task_type == "ps")
    thread.start()

    tf.logging.info(f"Started {task_type}:{task_id}")

    # "ps" tasks never terminate and therefore cannot be joined.
    if task_type == "ps":
        broadcast("stop/" + task)
        spec_from_kv(client.kv, "stop", num_workers, num_ps)
    else:
        thread.join()
        tf.logging.info(f"Stopped {task_type}:{task_id}")
        broadcast("stop/" + task)
        # Note that there is no synchronization happening here. In most
        # cases the chief/workers are independent and can terminate
        # without waiting for the rest of the cluster. This is not true
        # for "ps" tasks.

    if thread.exception() is not None:
        raise thread.exception()


if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--num-ps", type=int)
    parser.add_argument("--experiment-fn", type=str)

    try:
        experiment_fn = decode_fn(os.environ["EXPERIMENT_FN"])
    except KeyError:
        parser.error("EXPERIMENT_FN environment variable must be set")
    else:
        args = parser.parse_args()
        main(experiment_fn, args.num_workers, args.num_ps)
