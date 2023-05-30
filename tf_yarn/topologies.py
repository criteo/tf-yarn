import logging
from enum import Enum
from typing import Dict, Union, NamedTuple

import skein

GB = 2**10
MAX_MEMORY_CONTAINER = 48 * GB
MAX_VCORES_CONTAINER = 48
ALL_TASK_TYPES = {"chief", "worker", "ps", "evaluator", "tensorboard"}


logger = logging.getLogger(__name__)


class NodeLabel(Enum):
    """YARN node label expression.

    A task with a CPU label could be scheduled on any node, whereas
    a task with a GPU label, only on the one labeled with ``"gpu"``.
    """
    CPU = ""  # Default.
    GPU = "gpu"


class ContainerKey(NamedTuple):
    type: str
    id: int

    def to_kv_str(self):
        return f"{self.type}:{self.id}"

    @staticmethod
    def from_kv_str(kv_str: str):
        try:
            task_type, index = kv_str.split(":")
            return ContainerKey(task_type, int(index))
        except ValueError as e:
            logger.error(f"impossible to create ContainerKey from {kv_str} because {e}")


class ContainerTask(NamedTuple):
    type: str
    id: int
    nb_proc: int

    def to_container_key(self):
        return ContainerKey(self.type, self.id)

    def to_kv_str(self):
        return f"{self.type}:{self.id}:{self.nb_proc}"


class TaskSpec(object):
    __slots__ = ('_resources',
                 'instances',
                 'label',
                 'nb_proc_per_worker',
                 'tb_termination_timeout_seconds',
                 'tb_model_dir',
                 'tb_extra_args')

    def __init__(self,
                 memory: Union[int, str],
                 vcores: int,
                 instances: int = 1,
                 nb_proc_per_worker: int = 1,
                 label: NodeLabel = NodeLabel.CPU,
                 tb_termination_timeout_seconds: int = -1,
                 tb_model_dir: str = None,
                 tb_extra_args: str = None):
        self._resources = skein.model.Resources(memory, vcores)
        self.instances = instances
        self.nb_proc_per_worker = nb_proc_per_worker
        self.label = label
        self.tb_termination_timeout_seconds = tb_termination_timeout_seconds
        self.tb_model_dir = tb_model_dir
        self.tb_extra_args = tb_extra_args

    @property
    def memory(self) -> int:
        return self._resources.memory

    @memory.setter
    def memory(self, value: Union[int, str]):
        self._resources = skein.model.Resources(value, self._resources.vcores)

    @property
    def vcores(self) -> int:
        return self._resources.vcores

    @vcores.setter
    def vcores(self, value: int):
        self._resources = skein.model.Resources(self._resources.memory, value)


def _check_general_topology(task_specs: Dict[str, TaskSpec]) -> None:
    if not task_specs.keys() <= ALL_TASK_TYPES:
        raise ValueError(
            f"task_specs.keys() must be a subset of: {ALL_TASK_TYPES}")
    if task_specs["chief"].instances != 1:
        raise ValueError("exactly one 'chief' task is required")
    for task_type, spec in task_specs.items():
        if spec.memory > MAX_MEMORY_CONTAINER:
            raise ValueError(
                f"{task_type}: Can not demand more memory than "
                f"{MAX_MEMORY_CONTAINER} bytes for container")
        if spec.vcores > MAX_VCORES_CONTAINER:
            raise ValueError(
                f"{task_type}: Can not demand more vcores than "
                f"{MAX_VCORES_CONTAINER} for container")
        if spec.nb_proc_per_worker > spec.vcores:
            raise ValueError(
                f"{task_type}: Can not run more processes per instance than "
                f"vcore number of container ({spec.vcores})")


def _check_ps_topology(task_specs: Dict[str, TaskSpec]) -> None:
    _check_general_topology(task_specs)
    if task_specs["evaluator"].instances > 1 or task_specs["evaluator"].nb_proc_per_worker > 1:
        raise ValueError("no more than one 'evaluator' task is allowed")
    if task_specs["tensorboard"].instances > 1 or task_specs["tensorboard"].nb_proc_per_worker > 1:
        raise ValueError("no more than one 'tensorboard' task is allowed")
    if not task_specs["ps"].instances:
        raise ValueError(
            "task_specs must contain at least a single 'ps' task for "
            "multi-worker training")


def single_server_topology(
    memory: int = MAX_MEMORY_CONTAINER,
    vcores: int = MAX_VCORES_CONTAINER
) -> Dict[str, TaskSpec]:
    topology = {
        "chief": TaskSpec(memory=memory, vcores=vcores),
        "evaluator": TaskSpec(memory=memory, vcores=vcores),
        "tensorboard": TaskSpec(memory=memory, vcores=vcores)
    }
    _check_general_topology(topology)
    return topology


# Topology to use for TensorFlow distributed training with parameter server strategy
def ps_strategy_topology(
    nb_workers: int = 2,
    nb_ps: int = 1,
    memory: int = MAX_MEMORY_CONTAINER,
    vcores: int = MAX_VCORES_CONTAINER
) -> Dict[str, TaskSpec]:
    # TODO: compute num_ps from the model size and the number of
    # executors. See https://stackoverflow.com/a/46080567/262432.
    topology = {
        "chief": TaskSpec(memory=memory, vcores=vcores),
        "evaluator": TaskSpec(memory=memory, vcores=vcores),
        "worker": TaskSpec(memory=memory, vcores=vcores, instances=nb_workers),
        "ps": TaskSpec(memory=memory, vcores=vcores, instances=nb_ps),
        "tensorboard": TaskSpec(memory=memory, vcores=vcores, instances=1)
    }
    _check_ps_topology(topology)
    return topology
