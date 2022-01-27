from typing import Dict, Callable, Optional

from tf_yarn import client
from tf_yarn import topologies
from tf_yarn.metrics import Metrics
from tf_yarn.pytorch.experiment import PytorchExperiment


ExperimentFn = Callable[[], PytorchExperiment]


def run_on_yarn(
    experiment_fn: ExperimentFn,
    task_specs: Dict[str, topologies.TaskSpec],
    **kwargs
) -> Optional[Metrics]:
    if "custom_task_module" not in kwargs:
        kwargs["custom_task_module"] = "tf_yarn.pytorch.tasks.worker"
    return client.run_on_yarn(
        experiment_fn,
        task_specs,
        **kwargs
    )
