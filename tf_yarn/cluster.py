import json
import socket
import os

import typing
import skein
import tensorflow as tf

from . import _internal
from . import event


def aggregate_spec(client: skein.ApplicationClient,
                   all_tasks: typing.List[str]
                   ) -> typing.Dict[str, typing.List[str]]:
    spec: typing.Dict[str, typing.List[str]] = {}
    for task in sorted(all_tasks, key=lambda x: int(x.split(':', 1)[1])):
        sock_addr = event.wait(client, f"{task}/init")
        task_type, _task_id = task.split(":", 1)
        spec.setdefault(task_type, []).append(sock_addr)
    return spec


def get_task() -> str:
    return os.environ["SKEIN_CONTAINER_ID"].replace("_", ":", 1)


def get_task_description() -> typing.Tuple[str, int]:
    task = get_task()
    task_type, task_str = task.split(":", 1)
    task_id: int = int(task_str)
    return task_type, task_id


def start_cluster(
    host_port: typing.Tuple[str, int],
    client: skein.ApplicationClient,
    all_tasks: typing.List[str]
) -> typing.Dict[str, typing.List[str]]:
    # There is a race condition between acquiring a TCP port for
    # ``tf.train.Server``, and calling ``train_and_evaluate``.
    # There is no TensorFlow API to get rid of the race condition
    # completely, but the window of opportunity can be reduced by
    # preempting the server.
    # See https://github.com/tensorflow/tensorflow/issues/21492
    cluster_spec: typing.Dict = dict()
    host, port = host_port
    event.init_event(client, get_task(), f"{socket.gethostbyname(host)}:{port}")
    cluster_spec = aggregate_spec(client, all_tasks)
    return cluster_spec


def setup_tf_config(cluster_spec):
    # Note that "evaluator" does not need a cluster, and "ps" (!)
    # surprisingly does not follow the same code path as the rest
    # and spawns a server regardless of the "environment" value.
    task_type, task_id = get_task_description()
    _internal.xset_environ(TF_CONFIG=json.dumps({
        "cluster": cluster_spec,
        "environment": "google" if is_fake_google_env(task_type) else "",
        "task": {"type": task_type, "index": task_id},
    }))


def start_tf_server(
    cluster_spec: typing.Dict[str, typing.List[str]],
    session_config: tf.ConfigProto = None
) -> typing.Optional[tf.train.Server]:

    task_type, task_id = get_task_description()
    if is_fake_google_env(task_type) and cluster_spec:
        server = tf.train.Server(
            tf.train.ClusterSpec(cluster_spec),
            job_name=task_type,
            task_index=task_id,
            config=session_config,
            start=True)
        return server
    return None


def is_fake_google_env(task_type: str) -> bool:
    return task_type != "evaluator" and task_type != "ps"
