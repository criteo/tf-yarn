import logging
import os
import skein
import tensorflow as tf

from typing import Optional

from tf_yarn import Experiment, KerasExperiment
from tf_yarn.tasks import logging as tf_yarn_logging
tf_yarn_logging.setup()

from tf_yarn import (
    _task_commons,
    _internal,
    cluster,
    event,
    tensorboard
)

_logger = logging.getLogger(__name__)


def main() -> None:
    _task_commons._log_sys_info()
    task_type, task_id = cluster.get_task_description()
    task = cluster.get_task()
    client = skein.ApplicationClient.from_current()

    _task_commons._setup_container_logs(client)
    cluster_tasks = _task_commons._get_cluster_tasks(client)

    model_dir = os.getenv('TB_MODEL_DIR', "")
    if not model_dir:
        _logger.info("Read model_dir from estimator config")
        experiment = _task_commons._get_experiment(client)
        if isinstance(experiment, Experiment):
            model_dir = experiment.estimator.config.model_dir
        elif isinstance(experiment, KerasExperiment):
            model_dir = experiment.model_dir
        else:
            raise ValueError("experiment must be an Experiment or a KerasExperiment")

    _logger.info(f"Starting tensorboard on {model_dir}")

    thread = _internal.MonitoredThread(
        name=f"{task_type}:{task_id}",
        target=tensorboard.start_tf_board,
        args=(client, model_dir),
        daemon=True)
    thread.start()

    for cluster_task in cluster_tasks:
        event.wait(client, f"{cluster_task}/stop")

    timeout = tensorboard.get_termination_timeout()
    thread.join(timeout)

    event.stop_event(client, task, thread.exception)
    event.broadcast_container_stop_time(client, task)


if __name__ == "__main__":
    main()
