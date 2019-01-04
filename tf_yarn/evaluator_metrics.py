import sys
import tensorflow as tf
import time
import skein
import tf_yarn
from tf_yarn.experiment import Experiment
from typing import List, Dict, Tuple
import logging
from tf_yarn.cluster import get_task
import warnings

MONITORED_METRICS = {
    'awake_time_ratio': 'Awake/idle ratio',
    'eval_step_mean_duration': 'Eval step mean duration (in sec)',
    'last_training_step': 'Training step of last checkpoint',
    'nb_eval_steps': 'Number of evaluation steps done'
}

logger = logging.getLogger(__name__)


class EvaluatorMetricsLogger():
    def __init__(
        self,
        evaluator_list: List[str],
        app: skein.ApplicationClient,
        log_thresholds: Dict[str, Tuple[float, float]] = None
    ):
        self.last_metrics = {
            evaluator: {metric: None for metric in MONITORED_METRICS}
            for evaluator in evaluator_list
        }
        self.evaluator_list = evaluator_list
        self.app = app
        if log_thresholds:
            key_set = set(log_thresholds)
            self.log_thresholds = {
                key: (
                    log_thresholds[key][0] if log_thresholds[key][0] else 0,
                    log_thresholds[key][1] if log_thresholds[key][1] else sys.float_info.max
                )
                for key in key_set.intersection(MONITORED_METRICS)
            }
            diff = key_set.difference(MONITORED_METRICS)
            if len(diff) > 0:
                warnings.warn("The following evaluation metrics are not monitored: {diff}")
        else:
            self.log_thresholds = dict()

    def log(self):
        for evaluator in self.evaluator_list:
            cur_eval_stats = []
            for key, value in MONITORED_METRICS.items():
                stat = self.app.kv.get(f'{evaluator}/{key}', None)
                stat = float(stat.decode()) if stat else None
                if stat is not None and stat != self.last_metrics[evaluator][key]:
                    if key not in self.log_thresholds or\
                            (self.log_thresholds[key][0] <= stat <= self.log_thresholds[key][1]):
                        cur_eval_stats.append(f'{value}: {stat}')
                    self.last_metrics[evaluator][key] = stat
            if len(cur_eval_stats) > 0:
                logger.info(f'Statistics for {evaluator}: {" ".join(cur_eval_stats)}')


class EvalMonitorHook(tf.train.SessionRunHook):
    '''
    Hook to generate statistics about evaluator usage
    Usage: tf.estimator.EvalSpec(...,hooks=[Eval_analyzer_hook()])
    '''
    def __init__(self):
        self.client = skein.ApplicationClient.from_current()
        self.task = get_task()
        self.step_counter = 0
        self.eval_start_time = 0.0
        self.eval_step_dur_accu = 0.0
        self.start_time = time.time()

    def before_run(self, run_context):
        self.eval_start_time = time.time()
        return tf.train.SessionRunArgs(tf.train.get_global_step())

    def after_run(self, _run_context, run_values):
        self.step_counter += 1
        cur_time = time.time()
        self.eval_step_dur_accu += cur_time - self.eval_start_time
        self.broadcast('eval_step_mean_duration', str(self.eval_step_dur_accu / self.step_counter))
        self.broadcast(
            'awake_time_ratio',
            str(self.eval_step_dur_accu / (cur_time - self.start_time))
        )
        self.broadcast('nb_eval_steps', str(self.step_counter))
        self.broadcast('last_training_step', str(run_values.results))

    def broadcast(self, key: str, value: str):
        tf_yarn.event.broadcast(self.client, f'{self.task}/{key}', value)


def add_monitor_to_experiment(experiment: Experiment):
    monitored_eval_spec = experiment.eval_spec._replace(
        hooks=(EvalMonitorHook(), *experiment.eval_spec.hooks)
    )
    experiment = experiment._replace(eval_spec=monitored_eval_spec)
    return experiment
