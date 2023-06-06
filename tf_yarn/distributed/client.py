from typing import Dict, Callable, Optional

from tf_yarn import client
from tf_yarn import topologies
from tf_yarn.metrics import Metrics
from tf_yarn.client import ExperimentFn


def run_on_yarn(
    experiment_fn: ExperimentFn,
    task_specs: Dict[str, topologies.TaskSpec],
    **kwargs
) -> Optional[Metrics]:
    if "custom_task_module" not in kwargs:
        kwargs["custom_task_module"] = "tf_yarn.distributed.task"
    return client.run_on_yarn(
        experiment_fn,
        task_specs,
        **kwargs
    )
