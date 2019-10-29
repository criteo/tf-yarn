import cloudpickle
import logging
import tensorflow as tf

from tf_yarn.tasks import logging as tf_yarn_logging
tf_yarn_logging.setup()

from tf_yarn import _task_commons, _internal, event, cluster, constants

_logger = logging.getLogger(__name__)


def main() -> None:
    _task_commons._log_sys_info()
    task_type, task_id = cluster.get_task_description()
    with _internal.reserve_sock_addr() as host_port:
        client, cluster_spec, cluster_tasks = _task_commons._prepare_container(host_port)
        cluster.setup_tf_config(cluster_spec)
        tf_session_config = cloudpickle.loads(client.kv.wait(constants.KV_TF_SESSION_CONFIG))
        _logger.info(f"tf_server_conf {tf_session_config}")

    tf.contrib.distribute.run_standard_tensorflow_server()
    event.wait(client, "stop")


if __name__ == "__main__":
    main()
