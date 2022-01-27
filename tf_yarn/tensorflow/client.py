from typing import Union, Dict, Callable, Optional

from tf_yarn import client
from tf_yarn import topologies
from tf_yarn.metrics import Metrics
from tf_yarn.tensorflow.experiment import Experiment
from tf_yarn.tensorflow.keras_experiment import KerasExperiment
from tf_yarn.tensorflow.metrics import _add_monitor_to_experiment


ExperimentFn = Callable[[], Experiment]
KerasExperimentFn = Callable[[], KerasExperiment]

DEFAULT_TASK_SPEC = topologies.single_server_topology()


def run_on_yarn(
    experiment_fn: Union[ExperimentFn, KerasExperimentFn],
    task_specs: Dict[str, topologies.TaskSpec] = DEFAULT_TASK_SPEC,
    *args,
    **kwargs
) -> Optional[Metrics]:
    def _new_experiment_fn():
        return _add_monitor_to_experiment(experiment_fn())

    return client.run_on_yarn(
        _new_experiment_fn,
        task_specs,
        *args,
        **kwargs
    )
