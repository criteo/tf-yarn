import logging
import os
import sys
import time
import typing
from collections import defaultdict
from contextlib import contextmanager
from subprocess import check_output

import skein
import tensorflow as tf
from skein.model import ApplicationState

from ._internal import encode_fn, zip_inplace
from .env import Env


class Experiment(typing.NamedTuple):
    estimator: tf.estimator.Estimator
    train_spec: tf.estimator.TrainSpec
    eval_spec: tf.estimator.EvalSpec

    @property
    def config(self) -> tf.estimator.RunConfig:
        return self.estimator.config

    def __call__(self):
        return tf.estimator.train_and_evaluate(*self)


ExperimentFn = typing.Callable[[], Experiment]


class TaskSpec(typing.NamedTuple):
    memory: int
    vcores: int
    instances: int = 1


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
    task_specs
        Defines the resources to allocate for each task type. The keys
        must be a subset of ``"chief"``, ``"worker"``, ``"ps"``, and
        ``"evaluator"``. The minimal spec must contain at least
        ``"chief"``.
    env
        Defines the Python environment on the instances.
    """
    def __init__(
        self,
        task_specs: typing.Dict[str, TaskSpec],
        env: Env = Env.MINIMAL_CPU
    ) -> None:
        self.task_specs = defaultdict(lambda: TaskSpec.NONE, task_specs)
        self.env = env
        self.logger = logging.getLogger(self.__class__.__name__)

        all_task_types = {"chief", "worker", "ps", "evaluator"}
        if not task_specs.keys() <= all_task_types:
            raise ValueError(
                f"task_spec keys must be a subset of: {all_task_types}")

        # TODO: compute num_ps from the model size and the number of
        # executors. See https://stackoverflow.com/a/46080567/262432.
        assert self.task_specs["evaluator"].instances <= 1
        assert self.task_specs["chief"].instances == 1

    def __repr__(self):
        return f"SkeinCluster({self.task_specs}, {self.env})"

    __str__ = __repr__

    def run(
        self,
        experiment_fn: ExperimentFn,
        *,
        queue: str = "default",
        files: typing.Dict[str, str] = None
    ):
        """
        Run an experiment on YARN.

        Parameters
        ----------
        experiment_fn
            A function constructing the estimator alongside the train
            and eval specs.

        queue
            YARN queue to use.

        files
            Local files or directories to upload to the container.
            The keys are the target locations of the resources relative
            to the container root, while the values -- their
            corresponding local sources. Note that container root is
            appended to ``PYTHONPATH``. Therefore, any listed Python
            module a package is automatically importable.
        """
        env_name = self.env.name

        classpath = check_output([
            os.path.join(os.environ["HADOOP_HOME"], "bin", "hadoop"),
            "classpath",
            "--glob"
        ]).decode().strip()
        krb5_cc_name = os.environ["KRB5CCNAME"].replace("FILE:", "", 1)

        task_files = {
            env_name: self.env.create(),
            os.path.basename(krb5_cc_name): krb5_cc_name,
            __package__: zip_inplace(os.path.dirname(__file__), replace=True)
        }

        for target, source in (files or {}).items():
            assert target not in task_files
            task_files[target] = (
                zip_inplace(source) if os.path.isdir(source) else source
            )

        task_env = {
            # Make Python modules/packages passed via ``self.env.files``
            # importable.
            "PYTHONPATH": ".",
            "KRB5CCNAME": os.path.basename(krb5_cc_name),
            "HADOOP_HDFS_HOME": "/usr/lib/hadoop-hdfs",
            "CLASSPATH": classpath,
            "LD_LIBRARY_PATH": ":".join(
                [f"{os.environ['JAVA_HOME']}/jre/lib/amd64/server",
                 "/usr/lib/hadoop-criteo/hadoop/lib/native"]),
        }
        task_command = (
            f"{env_name}/bin/python -m tf_skein._dispatch_task "
            f"--num-ps={self.task_specs['ps'].instances} "
            f"--num-workers={self.task_specs['worker'].instances} "
            f"--experiment-fn={encode_fn(experiment_fn)}")

        services = {}
        for task_type, task_spec in self.task_specs.items():
            if task_spec is TaskSpec.NONE:
                continue

            services[task_type] = skein.Service(
                [task_command],
                skein.Resources(task_spec.memory, task_spec.vcores),
                instances=task_spec.instances,
                files=task_files,
                env=task_env)

        # TODO: experiment name?
        spec = skein.ApplicationSpec(services, queue=queue)
        security = skein.Security.from_new_directory(force=True)
        with skein.Client(security=security) as client:
            app_id = client.submit(spec)

            experiment = experiment_fn()
            self.logger.info(f"Starting training")
            # TODO: run TB automatically via ``tensorboard.program``.
            self.logger.info(
                f"Run ``tensorboard --logdir={experiment.config.model_dir}`` "
                "to monitor the training metrics in TensorBoard.")

            with _shutdown(client.connect(app_id)):
                while client.application_report(app_id).state not in [
                    ApplicationState.FINISHED,
                    ApplicationState.FAILED,
                    ApplicationState.KILLED
                ]:
                    time.sleep(30)

            # TODO: report failures, ideally giving links to the logs of
            # the failed containers.
            return experiment.estimator


@contextmanager
def _shutdown(app: skein.ApplicationClient):
    try:
        yield app
    finally:
        _exc_type, exc_value, _exc_tb = sys.exc_info()
        if isinstance(exc_value, (KeyboardInterrupt, SystemExit)):
            status = "KILLED"
        elif exc_value is not None:
            status = "FAILED"
        else:
            status = "SUCCEEDED"

        try:
            app.shutdown(status)
        except skein.exceptions.ConnectionError:
            pass  # Application already down.
