import json
import socket
from typing import List, Dict, Tuple, Optional

import skein
import tensorflow as tf

from tf_yarn import _internal
from tf_yarn import event
from tf_yarn._task_commons import get_task_key
from tf_yarn.topologies import ContainerTask


def aggregate_spec(client: skein.ApplicationClient,
                   all_tasks: List[ContainerTask]
                   ) -> Dict[str, List[str]]:
    spec: Dict[str, List[str]] = {}
    for task in sorted(all_tasks, key=lambda x: x.id):
        sock_addr = event.wait(client, f"{task.to_container_key().to_kv_str()}/init")
        spec.setdefault(task.type, []).append(sock_addr)
    return spec


def start_cluster(
    host_port: Tuple[str, int],
    client: skein.ApplicationClient,
    all_tasks: List[ContainerTask]
) -> Dict[str, List[str]]:
    # There is a race condition between acquiring a TCP port for
    # ``tf.train.Server``, and calling ``train_and_evaluate``.
    # There is no TensorFlow API to get rid of the race condition
    # completely, but the window of opportunity can be reduced by
    # preempting the server.
    # See https://github.com/tensorflow/tensorflow/issues/21492
    host, port = host_port
    event.init_event(client, get_task_key(), f"{socket.gethostbyname(host)}:{port}")
    cluster_spec = aggregate_spec(client, all_tasks)
    return cluster_spec


def setup_tf_config(cluster_spec: Dict[str, List[str]]):
    # Note that "evaluator" does not need a cluster, and "ps" (!)
    # surprisingly does not follow the same code path as the rest
    # and spawns a server regardless of the "environment" value.
    task_type, task_id = get_task_key()
    _internal.xset_environ(TF_CONFIG=json.dumps({
        "cluster": cluster_spec,
        "environment": "google" if _is_fake_google_env(task_type) else "",
        "task": {"type": task_type, "index": task_id},
    }))


def start_tf_server(
    cluster_spec: Dict[str, List[str]],
    session_config: tf.compat.v1.ConfigProto = None
) -> Optional[tf.distribute.Server]:

    task_key = get_task_key()
    if _is_fake_google_env(task_key.type) and cluster_spec:
        server = tf.distribute.Server(
            tf.train.ClusterSpec(cluster_spec),
            job_name=task_key.type,
            task_index=task_key.id,
            config=session_config,
            start=True)
        return server
    return None


def _is_fake_google_env(task_type: str) -> bool:
    return task_type != "evaluator" and task_type != "ps"
