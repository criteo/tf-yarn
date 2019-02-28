import dill
import tensorflow as tf

from . import event
from tf_yarn._task_commons import _prepare_container, _process_arguments
from . import cluster, KV_TF_SESSION_CONFIG


def main() -> None:
    task_type, task_id = cluster.get_task_description()
    client, cluster_spec, cluster_tasks = _prepare_container()
    cluster.setup_tf_config(cluster_spec)
    tf_session_config = dill.loads(client.kv.wait(KV_TF_SESSION_CONFIG))
    tf.logging.info(f"tf_server_conf {tf_session_config}")
    cluster.start_tf_server(cluster_spec, tf_session_config)
    event.wait(client, "stop")


if __name__ == "__main__":
    _process_arguments()
    main()
