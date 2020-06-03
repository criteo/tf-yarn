import logging
import skein
import tensorflow as tf
import os

from datetime import timedelta

from typing import (
    List,
    Optional,
    NamedTuple,
    Dict,
    Tuple
)

from tf_yarn import mlflow, cluster

logger = logging.getLogger(__name__)


class Metrics(NamedTuple):
    total_training_duration: Optional[timedelta]
    total_eval_duration: Optional[timedelta]
    container_duration: Dict[str, Optional[timedelta]]
    train_eval_time_per_node: Dict[str, Optional[timedelta]]

    def log_mlflow(self, n_try: int):
        content = ""
        for metric_name, value in self._asdict().items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if v:
                        formatted_key = mlflow.format_key(f"{metric_name}_{k}_{n_try}")
                        content = content + f"{formatted_key}: {v.total_seconds()} secs\n"
            else:
                if value:
                    formatted_key = mlflow.format_key(f"{metric_name}_{n_try}")
                    content = content + f"{formatted_key}: {value.total_seconds()} secs\n"

        mlflow.save_text_to_mlflow(content, "tf_yarn_duration_stats")


class OneShotMetricsLogger(NamedTuple):
    app: skein.ApplicationClient
    events: List[Tuple[str, str]]
    n_try: int = 0

    def log(self):
        new_events = [event for event in self.events if not self.__log(*event)]
        del self.events[:]
        self.events.extend(new_events)

    def __log(self, key, label):
        ret = False
        value = self.app.kv.get(key, None)
        if value:
            value = value.decode()
            logger.info(f"{label} {value}")
            mlflow.set_tag(f"{mlflow.format_key(key)}_{self.n_try}", value)
            ret = True
        return ret


class StepPerSecondHook(tf.estimator.StepCounterHook):

    def __init__(
        self,
        every_n_steps=100,
        every_n_secs=None,
        output_dir=None,
        summary_writer=None
    ):
        tf.estimator.StepCounterHook.__init__(
            self,
            every_n_steps=every_n_steps,
            every_n_secs=every_n_secs,
            output_dir=output_dir,
            summary_writer=summary_writer
        )

    def _log_and_record(self, elapsed_steps: int, elapsed_time: float, global_step: int):
        if cluster.is_chief():
            steps_per_sec = elapsed_steps / elapsed_time
            mlflow.log_metric(f"steps_per_sec_{cluster.n_try()}", steps_per_sec, step=global_step)


def is_event_file(filename):
    return os.path.basename(filename).startswith('events.out')


def gen_events_iterator(model_path):
    event_file = next((filename for filename in tf.compat.v1.gfile.ListDirectory(model_path)
                       if is_event_file(filename)))
    assert isinstance(event_file, str)
    return tf.compat.v1.train.summary_iterator(os.path.join(model_path, event_file))


def get_all_metrics(model_path):
    events = gen_events_iterator(model_path)
    dataframe = {
        'step': list(),
        'name': list(),
        'value': list()
    }
    for event in events:
        summary = event.summary
        if summary:
            for value in summary.value:
                if value.simple_value:
                    dataframe['step'].append(event.step)
                    dataframe['name'].append(value.tag)
                    dataframe['value'].append(value.simple_value)
    return dataframe
