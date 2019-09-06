import logging
import skein
import tensorflow as tf

from datetime import timedelta

from typing import (
    List,
    Optional,
    NamedTuple,
    Dict
)

from tf_yarn import mlflow, cluster

logger = logging.getLogger(__name__)


class Metrics(NamedTuple):
    total_training_duration: Optional[timedelta]
    total_eval_duration: Optional[timedelta]
    container_duration: Dict[str, Optional[timedelta]]
    train_eval_time_per_node: Dict[str, Optional[timedelta]]

    def log_mlflow(self):
        for metric_name, value in self._asdict().items():
            if isinstance(value, dict):
                mlflow.log_params({mlflow.format_key(f"{metric_name}_{k}"): v
                                   for k, v in value.items()})
            else:
                mlflow.log_param(metric_name, value)


class OneShotMetricsLogger(object):

    def __init__(
            self,
            app: skein.ApplicationClient,
            keys_per_task: Dict[str, List[str]]
    ):
        self.app = app
        self.metrics: List[str] = self.__init_metrics(keys_per_task)

    def __init_metrics(self, keys_per_task):
        metrics = []
        if keys_per_task is not None:
            for task, keys in keys_per_task.items():
                for key in keys:
                    metrics.append(f"{task}/{key}")
        return metrics

    def log(self):
        self.metrics = [metric for metric in self.metrics if not self.__log(metric)]

    def __log(self, metric):
        ret = False
        value = self.app.kv.get(metric, None)
        if value is not None:
            value = value.decode()
            logger.info(f"{value}")
            mlflow.set_tag(mlflow.format_key(metric), value)
            ret = True
        return ret


class StepPerSecondHook(tf.train.StepCounterHook):

    def __init__(
        self,
        every_n_steps=100,
        every_n_secs=None,
        output_dir=None,
        summary_writer=None
    ):
        tf.train.StepCounterHook.__init__(
            self,
            every_n_steps=every_n_steps,
            every_n_secs=every_n_secs,
            output_dir=output_dir,
            summary_writer=summary_writer
        )

    def _log_and_record(self, elapsed_steps: int, elapsed_time: float, global_step: int):
        if cluster.is_chief():
            steps_per_sec = elapsed_steps / elapsed_time
            mlflow.log_metric("steps_per_sec", steps_per_sec, step=global_step)
