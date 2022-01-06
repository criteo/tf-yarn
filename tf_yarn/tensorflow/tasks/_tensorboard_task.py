import logging
import os

import skein

from tf_yarn.tensorflow import Experiment, KerasExperiment
from tf_yarn.tensorflow.tasks.tf_task_common import _log_sys_info
from tf_yarn._task_commons import (
    setup_logging, get_task, get_task_description, _get_cluster_tasks,
    _setup_container_logs, _get_experiment
)
setup_logging()

from tf_yarn import (
    _internal,
    event,
)
from tf_yarn import tensorboard

_logger = logging.getLogger(__name__)


# TODO: this taks could be agnostic of the ML framework
# We only need to provide the model directory
def main() -> None:
    _log_sys_info()
    task_type, task_id = get_task_description()
    task = get_task()
    client = skein.ApplicationClient.from_current()

    _setup_container_logs(client)
    cluster_tasks = _get_cluster_tasks(client)

    model_dir = os.getenv('TB_MODEL_DIR', "")
    if not model_dir:
        _logger.info("Read model_dir from estimator config")
        experiment = _get_experiment(client)
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
