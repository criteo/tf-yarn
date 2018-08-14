import argparse
import json
import logging
import os
import sys

import skein
import tensorflow as tf

from . import ExperimentFn
from ._internal import (
    MonitoredThread,
    reserve_sock_addr,
    decode_fn,
    spec_from_kv,
    xset_environ
)


def main(
    experiment_fn: ExperimentFn,
    num_workers: int,
    num_ps: int
) -> None:
    def broadcast(key: str, value: str = ""):
        tf.logging.info(f"Broadcasting {key} = {value!r}")
        client.kv[key] = value.encode()

    tf.logging.info("Python " + sys.version)
    tf.logging.info("Skein " + skein.__version__)
    tf.logging.info(f"TensorFlow {tf.GIT_VERSION} {tf.VERSION}")

    task = os.environ["SKEIN_CONTAINER_ID"]
    task_type, task_id = task.split("_", 1)
    task_id = int(task_id)
    client = skein.ApplicationClient.from_current()

    # There is a race condition between acquiring a TPC port for the
    # ``tf.train.Server``, and calling ``train_and_evaluate``.
    # There is no TensorFlow API to get rid of the race condition
    # completely, but the window of opportunity can be reduced by
    # preempting the server.
    # See https://github.com/tensorflow/tensorflow/issues/21492
    with reserve_sock_addr() as (host, port):
        broadcast("init/" + task, f"{host}:{port}")
        spec = spec_from_kv(client.kv, "init", num_workers, num_ps)

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

    if task_type == "ps":
        broadcast("stop/" + task)
    else:
        thread.join()
        tf.logging.info(f"Stopped {task_type}:{task_id}")
        broadcast("stop/" + task)

    # The following (pessimistically) assumes the communication graph
    # between the tasks is complete. This means that each task has to
    # wait for all other tasks before exiting. This might not be true
    # in the presence of ``device_filters``.
    spec_from_kv(client.kv, "stop", num_workers, num_ps)
    if thread.exception() is not None:
        raise thread.exception() from None


if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--num-ps", type=int)
    parser.add_argument("--experiment-fn", type=decode_fn)
    main(**vars(parser.parse_args()))
