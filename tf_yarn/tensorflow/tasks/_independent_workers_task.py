import logging

from tf_yarn._task_commons import setup_logging, _get_experiment
setup_logging()

from tf_yarn import _internal
from tf_yarn.tensorflow import Experiment, KerasExperiment
from tf_yarn.tensorflow.tasks.tf_task_common import (
    _execute_dispatched_function, _shutdown_container, _prepare_container,
    get_task_description, _log_sys_info
)
from tf_yarn.tensorflow import cluster

_logger = logging.getLogger(__name__)


def main() -> None:
    _log_sys_info()
    task_type, task_id = get_task_description()
    with _internal.reserve_sock_addr() as host_port:
        client, cluster_spec, cluster_tasks = _prepare_container(host_port)
        # Variable TF_CONFIG must be set before instantiating
        # the estimator to train in a distributed way
        cluster.setup_tf_config(cluster_spec)
        experiment = _get_experiment(client)
        if isinstance(experiment, Experiment):
            session_config = experiment.config.session_config
        elif isinstance(experiment, KerasExperiment):
            raise ValueError("KerasExperiment using parameter strategy is unsupported")
        else:
            raise ValueError("experiment must be an Experiment or a KerasExperiment")
        _logger.info(f"Starting server {task_type}:{task_id}")

    cluster.start_tf_server(cluster_spec, session_config)
    thread = _execute_dispatched_function(client, experiment)

    # "ps" tasks do not terminate by themselves. See
    # https://github.com/tensorflow/tensorflow/issues/4713.
    if task_type not in ['ps']:
        thread.join()
        _logger.info(f"{task_type}:{task_id} {thread.state}")

    _shutdown_container(client, cluster_tasks, session_config, thread)


if __name__ == "__main__":
    main()
