import cloudpickle
import tensorflow as tf

from . import event
from tf_yarn._task_commons import _prepare_container, _process_arguments
from tf_yarn._internal import reserve_sock_addr
from . import cluster, KV_TF_SESSION_CONFIG


def main() -> None:
    task_type, task_id = cluster.get_task_description()
    with reserve_sock_addr() as host_port:
        client, cluster_spec, cluster_tasks = _prepare_container(host_port)
        cluster.setup_tf_config(cluster_spec)
        tf_session_config = cloudpickle.loads(client.kv.wait(KV_TF_SESSION_CONFIG))
        tf.logging.info(f"tf_server_conf {tf_session_config}")

    tf.contrib.distribute.run_standard_tensorflow_server()
    event.wait(client, "stop")


if __name__ == "__main__":
    _process_arguments()
    main()
