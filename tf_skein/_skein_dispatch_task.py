import argparse
import json
import os
import time
from contextlib import closing

import skein
import tensorflow as tf

from ._internal import (
    iter_available_sock_addrs,
    _spec_from_kv,
    MonitoredThread,
    decode_fn,
    set_env)
from .cluster import ExperimentFn, ConfigFn


class ApplicationClient(skein.ApplicationClient):
    @property
    def current_container(self):
        # TODO: ensure new skein on the executors.
        return next(c for c in self.containers()
                    if c.yarn_container_id == os.environ["CONTAINER_ID"])


def main(
    task_type: str,
    config_fn: ConfigFn,
    experiment_fn: ExperimentFn,
    num_workers: int,
    num_ps: int
):
    client = ApplicationClient.from_current()
    with closing(iter_available_sock_addrs()) as it:
        task_id = client.current_container.instance
        task = f"{task_type}:{task_id}"
        client.kv[task] = sock_addr = next(it)

        # This blocks waiting for other tasks to register.
        spec = _spec_from_kv(client.kv, num_workers, num_ps)

    # Preempt to ensure all tasks in the cluster are ready to
    # accept incoming traffic by the time we create the training
    # session. Note that "evaluator" does not need a cluster,
    # and (for unknown reasons) "ps" does not follow the same
    # code path as the rest and spawns a server regardless of
    # the "environment" value.
    fake_google_env = task_type != "evaluator" and task_type != "ps"
    tf_config = json.dumps({
        "cluster": spec,
        "task": {"type": task_type, "index": task_id},
        "environment": "google" if fake_google_env else ""
    })
    with set_env(TF_CONFIG=tf_config):
        config = config_fn()

    if fake_google_env:
        # XXX this is not immune to a race condition.
        tf.train.Server(
            config.cluster_spec,
            job_name=config.task_type,
            task_index=config.task_id,
            config=config.session_config,
            start=True)

    task = f"{task_type}:{task_id}"
    thread = MonitoredThread(
        name=task,
        target=experiment_fn(config),
        # "ps" tasks do not terminate by themselves. See
        # https://github.com/tensorflow/tensorflow/issues/4713
        daemon=task_type == "ps")
    thread.start()

    tf.logging.info(f"Started {task}")

    # "ps" tasks never terminate and therefore cannot be joined.
    if task_type != "ps":
        thread.join()
        if thread.exception():
            raise thread.exception()

    # TODO: explain that chief _creates_ a key and unblocks others.
    if task_type == "chief":
        client.kv["stopped"] = task + ";"
    else:
        while True:
            stopped = client.kv.wait("stopped")
            if task + ";" in stopped:
                break

            client.kv["stopped"] = stopped + task + ";"

    while True:
        stopped = client.kv.wait("stopped")
        if stopped.count(";") == num_workers + num_ps + 2:
            break
        time.sleep(30)

    return sock_addr, task


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--num-ps", type=int)
    parser.add_argument("--config-fn", type=str)
    parser.add_argument("--experiment-fn", type=str)
    parser.add_argument(
        "task_type",
        choices=["ps", "worker", "chief", "evaluator"])

    args = parser.parse_args()
    main(
        args.task_type,
        decode_fn(args.config_fn),
        decode_fn(args.experiment_fn),
        args.num_workers,
        args.num_ps)
