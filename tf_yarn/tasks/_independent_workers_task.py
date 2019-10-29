import logging
import os
import sys

import tensorflow as tf

from tf_yarn.tasks import logging as tf_yarn_logging
tf_yarn_logging.setup()

from tf_yarn import cluster, _task_commons, _internal

_logger = logging.getLogger(__name__)


def main() -> None:
    _task_commons._log_sys_info()
    task_type, task_id = cluster.get_task_description()
    with _internal.reserve_sock_addr() as host_port:
        client, cluster_spec, cluster_tasks = _task_commons._prepare_container(host_port)
        # Variable TF_CONFIG must be set before instantiating
        # the estimator to train in a distributed way
        cluster.setup_tf_config(cluster_spec)
        experiment = _task_commons._get_experiment(client)
        run_config = experiment.config
        _logger.info(f"Starting server {task_type}:{task_id}")

    cluster.start_tf_server(cluster_spec, run_config.session_config)
    thread = _task_commons._execute_dispatched_function(client, experiment)

    # "ps" tasks do not terminate by themselves. See
    # https://github.com/tensorflow/tensorflow/issues/4713.
    if task_type not in ['ps']:
        thread.join()
        _logger.info(f"{task_type}:{task_id} {thread.state}")

    _task_commons._shutdown_container(client, cluster_tasks, run_config, thread)


if __name__ == "__main__":
    main()
