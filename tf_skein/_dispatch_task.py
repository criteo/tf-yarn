import argparse
import json
import logging
import os
from contextlib import closing

import skein
import tensorflow as tf

from ._internal import (
    KVBarrier,
    MonitoredThread,
    iter_available_sock_addrs,
    decode_fn,
    xset_environ
)
from .cluster import ExperimentFn


def main(
    experiment_fn: ExperimentFn,
    num_workers: int,
    num_ps: int
):
    client = skein.ApplicationClient.from_current()
    current_container = next(
        c for c in client.get_containers()
        if c.yarn_container_id == os.environ["CONTAINER_ID"])

    task_type = current_container.service_name
    task_id = current_container.instance
    task = f"{task_type}:{task_id}"

    with closing(iter_available_sock_addrs()) as it:
        init_barrier = KVBarrier(client.kv, "init", num_workers, num_ps)
        sock_addr = next(it)
        spec = init_barrier.wait(task, sock_addr)

    # Preempt to ensure all tasks in the cluster are ready to
    # accept incoming traffic by the time we create the training
    # session. Note that "evaluator" does not need a cluster,
    # and (for unknown reasons) "ps" does not follow the same
    # code path as the rest and spawns a server regardless of
    # the "environment" value.
    fake_google_env = task_type != "evaluator" and task_type != "ps"
    xset_environ(TF_CONFIG=json.dumps({
        "cluster": spec,
        "task": {"type": task_type, "index": task_id},
        "environment": "google" if fake_google_env else ""
    }))

    experiment = experiment_fn()
    config = experiment.config
    assert config.task_type == task_type
    assert config.task_id == task_id

    if fake_google_env:
        # XXX at this point the socket has already been closed.
        #     Therefore, there is a potential race with concurrent
        #     applications running on the same node. However,
        #     ``tf.train.Server`` provides no API for wrapping an
        #     existing TCP socket.
        tf.train.Server(
            config.cluster_spec,
            job_name=config.task_type,
            task_index=config.task_id,
            config=config.session_config,
            start=True)

    thread = MonitoredThread(
        name=task,
        target=experiment,
        # "ps" tasks do not terminate by themselves. See
        # https://github.com/tensorflow/tensorflow/issues/4713
        daemon=task_type == "ps")
    thread.start()

    tf.logging.info(f"Started {task}")

    # "ps" tasks never terminate and therefore cannot be joined.
    if task_type != "ps":
        thread.join()
        if thread.exception() is not None:
            raise thread.exception()

    stop_barrier = KVBarrier(client.kv, "stop", num_workers, num_ps)
    stop_barrier.wait(task)


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
