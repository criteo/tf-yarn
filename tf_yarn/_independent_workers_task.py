import tensorflow as tf

from tf_yarn._task_commons import (
    _prepare_container, _execute_dispatched_function,
    _shutdown_container, _process_arguments, _get_experiment
)
from tf_yarn._internal import reserve_sock_addr
from . import cluster


def main() -> None:
    task_type, task_id = cluster.get_task_description()
    with reserve_sock_addr() as host_port:
        client, cluster_spec, cluster_tasks = _prepare_container(host_port)
        # Variable TF_CONFIG must be set before instantiating
        # the estimator to train in a distributed way
        cluster.setup_tf_config(cluster_spec)
        experiment = _get_experiment(client)
        run_config = experiment.config
        tf.logging.info(f"Starting server {task_type}:{task_id}")

    cluster.start_tf_server(cluster_spec, run_config.session_config)
    thread = _execute_dispatched_function(client, experiment)

    # "ps" tasks do not terminate by themselves. See
    # https://github.com/tensorflow/tensorflow/issues/4713.
    # Tensorboard is terminated after all other tasks in _shutdown_container
    if task_type not in ['ps', 'tensorboard']:
        thread.join()
        tf.logging.info(f"{task_type}:{task_id} {thread.state}")

    _shutdown_container(client, cluster_tasks, run_config, thread)


if __name__ == "__main__":
    _process_arguments()
    main()
