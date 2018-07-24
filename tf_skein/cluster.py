import abc
import json
import logging
import os
import shutil
import sys
import time
import traceback
import typing
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing, contextmanager
from subprocess import check_output

import dill
import skein
import tensorflow as tf
from skein.model import ApplicationState

from .env import Env
from ._internal import (
    iter_available_sock_addrs,
    _spec_from_iter,
    encode_fn,
    xset_environ
)


class Experiment(typing.NamedTuple):
    estimator: tf.estimator.Estimator
    train_spec: tf.estimator.TrainSpec
    eval_spec: tf.estimator.EvalSpec

    def __call__(self):
        return tf.estimator.train_and_evaluate(*self)


ConfigFn = typing.Callable[[], tf.estimator.RunConfig]
ExperimentFn = typing.Callable[[tf.estimator.RunConfig], Experiment]


class Cluster(abc.ABC):
    @property
    def logger(self):
        return logging.getLogger(self.__class__.__name__)

    @abc.abstractmethod
    def run(self, config_fn: ConfigFn, experiment_fn: ExperimentFn):
        """Train a model in this cluster."""
        raise NotImplementedError


class LocalCluster(Cluster):
    """Single-node cluster.

    By default, the cluster is allocated with just a "chief" worker,
    and an "evaluator", meaning that all the training happens within
    the same process. It is possible to introduce other "worker"
    and "ps" tasks, if needed, but this is likely to degrade
    performance.
    """
    @classmethod
    def allocate(cls, *, num_workers=0, num_ps=0):
        with closing(iter_available_sock_addrs()) as it:
            spec = {
                "chief": [next(it)],
            }

            for _ in range(num_workers):
                spec.setdefault("worker", []).append(next(it))
            for _ in range(num_ps):
                spec.setdefault("ps", []).append(next(it))
            return cls(spec)

    def __init__(self, spec: typing.Dict[str, typing.List[str]]):
        self.spec = spec

    def __repr__(self):
        return f"LocalCluster({self.spec})"

    __str__ = __repr__

    def run(self, config_fn: ConfigFn, experiment_fn: ExperimentFn):
        tasks = list(self.spec.items())
        tasks.append(("evaluator", [None]))
        with ThreadPoolExecutor() as executor:
            futures = []
            for task, hosts in tasks:
                for idx, _host in enumerate(hosts):
                    tf_config = json.dumps({
                        "cluster": self.spec,
                        "task": {"type": task, "index": idx}
                    })
                    with xset_environ(TF_CONFIG=tf_config):
                        config = config_fn()
                    futures.append(executor.submit(experiment_fn(config)))

            for future in as_completed(futures):
                exc_value = future.exception()
                if exc_value:
                    # Simply re-throwing the exception keeps the executor
                    # waiting for the remaining tasks in an exit hook.
                    traceback.print_exception(
                        type(exc_value),
                        exc_value,
                        exc_value.__traceback__)
                    os._exit(-1)


class TaskSpec(typing.NamedTuple):
    memory: int
    vcores: int
    instances: int = 1


#: A "dummy" ``TaskSpec``.
TaskSpec.NONE = TaskSpec(0, 0, 0)


class YARNCluster(Cluster):
    """Multi-node cluster running on Skein.

    The implementation schedules each distributed TensorFlow task on
    a dedicated PySpark executor. Roughly, it proceeds as follows:

    1. Create a "fake" RDD with one partition per executor.
    2. On each executor find an available TCP port and communicate the
       resulting socket address (host/port pair) to other executors
       using the "init" barrier. This is a synchronization point
       which ensures that all tasks in the cluster are ready to
       talk over the network before the Estimator machinery attempts
       to initialize a `tf.train.MonitoredSession`.
    3. Reconstruct the cluster spec from the list of socket addresses
       accumulated by the barrier, and preempt a TensorFlow server.
    4. Start the training and synchronize on the "stop" barrier.
       The barrier compensates for the fact that "ps" tasks never
       terminate, and therefore should be killed, once all other
       tasks are finished.
    """
    env = Env(
        name=__package__,
        packages=[
            "dill==" + dill.__version__,
            "git+http://github.com/jcrist/skein",
            "tensorflow==" + tf.__version__
        ])

    def __init__(self, task_specs: typing.Dict[str, TaskSpec]):
        self.task_specs = defaultdict(lambda: TaskSpec.NONE, task_specs)

        # TODO: compute num_ps from the model size and the number of
        # executors. See https://stackoverflow.com/a/46080567/262432.
        assert self.task_specs["evaluator"].instances <= 1
        assert self.task_specs["chief"].instances <= 1

    def __repr__(self):
        return f"SkeinCluster({self.task_specs})"

    __str__ = __repr__

    def run(self, config_fn: ConfigFn, experiment_fn: ExperimentFn):
        env_name = self.env.name
        env_path = self.env.create()

        # TODO: allow to pass extra files and env. variables.g

        classpath = check_output([
            os.path.join(os.environ["HADOOP_HOME"], "bin", "hadoop"),
            "classpath",
            "--glob"
        ]).decode().strip()
        krb5_cc_name = os.environ["KRB5CCNAME"].replace("FILE:", "", 1)

        task_files = {
            env_name: env_path,
            os.path.basename(krb5_cc_name): krb5_cc_name
        }

        task_env = {
            "KRB5CCNAME": os.path.basename(krb5_cc_name),
            "HADOOP_HDFS_HOME": "/usr/lib/hadoop-hdfs",
            "CLASSPATH": classpath,
            "LD_LIBRARY_PATH": ":".join(
                [f"{os.environ['JAVA_HOME']}/jre/lib/amd64/server",
                 "/usr/lib/hadoop-criteo/hadoop/lib/native"]),
        }
        task_command = (
            # TODO: remove env_name/env_name.
            # TODO: make sure tf_skein is part of the env.
            f"{env_name}/{env_name}/bin/python -m tf_skein._dispatch_task "
            f"--num-ps={self.task_specs['ps'].instances} "
            f"--num-workers={self.task_specs['worker'].instances} "
            f"--config-fn={encode_fn(config_fn)} "
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

        spec = skein.ApplicationSpec(services)
        security = skein.Security.from_new_directory(force=True)
        with skein.Client(security=security) as client:
            app_id = client.submit(spec)

            # TODO: run TB automatically via ``tensorboard.program``.

            self.logger.info(f"Starting training")
            self.logger.info(
                f"Run ``tensorboard --logdir={config_fn().model_dir}`` "
                "to monitor the training metrics in TensorBoard.")

            with _shutdown(client.connect(app_id)):
                while client.application_report(app_id).state not in [
                    ApplicationState.FINISHED,
                    ApplicationState.FAILED,
                    ApplicationState.KILLED
                ]:
                    time.sleep(30)


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
