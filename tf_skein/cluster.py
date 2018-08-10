import logging
import os
import time
import typing
from collections import defaultdict, ChainMap
from enum import Enum

import skein
import tensorflow as tf

from ._criteo import get_default_env_vars
from ._internal import encode_fn, zip_inplace
from .env import PyEnv

logger = logging.getLogger(__name__)


class Experiment(typing.NamedTuple):
    estimator: tf.estimator.Estimator
    train_spec: tf.estimator.TrainSpec
    eval_spec: tf.estimator.EvalSpec

    @property
    def config(self) -> tf.estimator.RunConfig:
        return self.estimator.config


ExperimentFn = typing.Callable[[], Experiment]


class TaskFlavor(Enum):
    """The flavor values are YARN node label expressions.

    That is, a task with a CPU flavor can be scheduled on any node,
    whereas a task with a GPU flavor, only on the one labeled with
    ``"gpu"``.
    """
    CPU = ""     # Default.
    GPU = "gpu"


NodeLabelFn = typing.Callable[[TaskFlavor], str]


class TaskSpec(typing.NamedTuple):
    memory: int
    vcores: int
    instances: int = 1
    flavor: TaskFlavor = TaskFlavor.CPU


#: A "dummy" ``TaskSpec``.
TaskSpec.NONE = TaskSpec(0, 0, 0)


class YARNCluster:
    """Multi-node cluster running on Skein.

    The implementation allocates a service with the requested number
    of instances for each distributed TensorFlow task type. Each
    instance runs ``_dispatch_task`` which roughly does the following.

    1. Find an an available TCP port and communicate the resulting
       socket address (host/port pair) to other instances using the
       "init" barrier. This is a synchronization point which ensures
       that all tasks in the cluster are ready to talk over the
       network before the Estimator machinery attempts to initialize
       a `tf.train.MonitoredSession`.
    2. Reconstruct the cluster spec from the list of socket addresses
       accumulated by the barrier, and preempt a TensorFlow server.
    3. Start the training and synchronize on the "stop" barrier.
       The barrier compensates for the fact that "ps" tasks never
       terminate, and therefore should be killed, once all other
       tasks are finished.

    Parameters
    ----------
    pyenv : PyEnv
        The Python environment to deploy on the containers.

    files : dict
        Local files or directories to upload to the container.
        The keys are the target locations of the resources relative
        to the container root, while the values -- their
        corresponding local sources. Note that container root is
        appended to ``PYTHONPATH``. Therefore, any listed Python
        module a package is automatically importable.

    env_vars : dict
        Environment variables to forward to the containers.
    """
    def __init__(
        self,
        pyenv: PyEnv = PyEnv.MINIMAL,
        files: typing.Dict[str, str] = None,
        env_vars: typing.Dict[str, str] = None
    ) -> None:
        self.pyenv = pyenv
        self.files = files or {}
        self.env_vars = ChainMap(env_vars or {}, get_default_env_vars())

    def __repr__(self) -> str:
        return f"SkeinCluster(env={self.pyenv})"

    __str__ = __repr__

    def run(
        self,
        experiment_fn: ExperimentFn,
        *,
        task_specs: typing.Dict[str, TaskSpec],
        queue: str = "default"
    ) -> None:
        """
        Run an experiment on YARN.

        Parameters
        ----------
        experiment_fn
            A function constructing the estimator alongside the train
            and eval specs.

        task_specs
            Resources to allocate for each task type. The keys
            must be a subset of ``"chief"``, ``"worker"``, ``"ps"``, and
            ``"evaluator"``. The minimal spec must contain at least
            ``"chief"``.

        queue
            YARN queue to use.
        """
        # TODO: compute num_ps from the model size and the number of
        # executors. See https://stackoverflow.com/a/46080567/262432.
        task_specs = defaultdict(lambda: TaskSpec.NONE, task_specs)
        _check_task_specs(task_specs)

        task_files = {
            self.pyenv.name: self.pyenv.create(),
            __package__: zip_inplace(os.path.dirname(__file__)),
        }

        for target, source in self.files.items():
            assert target not in task_files
            task_files[target] = (
                zip_inplace(source, replace=True)
                if os.path.isdir(source)
                else source
            )

        task_env = {
            **self.env_vars,
            "EXPERIMENT_FN": encode_fn(experiment_fn),
            # Make Python modules/packages passed via ``self.env.files``
            # importable.
            "PYTHONPATH": ".:" + self.env_vars.get("PYTHONPATH", ""),
        }

        # TODO: use internal PyPI for CPU-optimized TF.
        pyenvs = {
            TaskFlavor.CPU: self.pyenv.extended_with(
                self.pyenv.name + "_cpu",
                packages=["tensorflow"]),
            TaskFlavor.GPU: self.pyenv.extended_with(
                self.pyenv.name + "_gpu",
                packages=["tensorflow-gpu"])
        }

        services = {}
        for task_type, task_spec in list(task_specs.items()):
            if task_spec is TaskSpec.NONE:
                continue

            pyenv = pyenvs[task_spec.flavor]
            task_command = (
                f"{pyenv.name}/bin/python -m tf_skein._dispatch_task "
                f"--num-ps={task_specs['ps'].instances} "
                f"--num-workers={task_specs['worker'].instances} "
            )

            services[task_type] = skein.Service(
                [task_command],
                skein.Resources(task_spec.memory, task_spec.vcores),
                instances=task_spec.instances,
                node_label=task_spec.flavor.value,
                files={**task_files, pyenv.name: pyenv.create()},
                env=task_env)

        # TODO: experiment name?
        spec = skein.ApplicationSpec(
            services,
            queue=queue,  # TODO vvv generalize.
            name_nodes=["hdfs://prod-pa4", "hdfs://preprod-pa4"])
        with skein.Client() as client:
            logger.info(f"Submitting experiment to {queue} queue")
            app_id = client.submit(spec)
            final_status = _await_termination(client, app_id)
            logger.info(
                f"Application {app_id} finished with status {final_status}")
            # TODO: report per-container status via KV.


def _check_task_specs(task_specs):
    all_task_types = {"chief", "worker", "ps", "evaluator"}
    if not task_specs.keys() <= all_task_types:
        raise ValueError(
            f"task_specs.keys() must be a subset of: {all_task_types}")

    if task_specs["chief"].instances != 1:
        raise ValueError("exactly one 'chief' task is required")
    if task_specs["evaluator"].instances > 1:
        raise ValueError("no more than one 'evaluator' task is allowed")
    if task_specs["worker"].instances > 0 and not task_specs["ps"].instances:
        raise ValueError(
            "task_specs must contain at least a single 'ps' task for "
            "multi-worker training")


def _await_termination(
    client: skein.Client,
    app_id: str,
    poll_every_secs: int = 10
) -> skein.core.FinalStatus:
    # Ensure SIGINT is not masked to enable kill on C-c.
    import signal
    signal.signal(signal.SIGINT, signal.default_int_handler)

    try:
        while True:
            report = client.application_report(app_id)
            final_status = report.final_status
            # TODO: log status on each tick?
            if final_status != "undefined":
                return final_status

            time.sleep(poll_every_secs)
    except (KeyboardInterrupt, SystemExit):
        client.kill_application(app_id)
        return skein.core.FinalStatus.KILLED
