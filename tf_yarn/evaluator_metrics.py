import logging
import sys
import time
import warnings
from typing import List, Dict, Tuple

import skein

from tf_yarn import mlflow
from tf_yarn.topologies import ContainerTask

MONITORED_METRICS = {
    'awake_time_ratio': 'Awake/idle ratio',
    'eval_step_mean_duration': 'Eval step mean duration (in sec)',
    'last_training_step': 'Training step of last checkpoint',
    'nb_eval_steps': 'Number of evaluation steps done'
}

logger = logging.getLogger(__name__)


class EvaluatorMetricsLogger:
    def __init__(
        self,
        evaluator_list: List[ContainerTask],
        app: skein.ApplicationClient,
        log_thresholds: Dict[str, Tuple[float, float]] = None,
        n_try: int = 0
    ):
        self.last_metrics = {
            evaluator.to_container_key(): {metric: None for metric in MONITORED_METRICS}
            for evaluator in evaluator_list
        }
        self.evaluator_list = evaluator_list
        self.app = app
        self.n_try = n_try
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
            evaluator_key = evaluator.to_container_key()
            cur_eval_stats = []
            for metric_key, value in MONITORED_METRICS.items():
                stat = self.app.kv.get(f'{evaluator_key.to_kv_str()}/{metric_key}', None)
                stat = float(stat.decode()) if stat else None
                if stat is not None and stat != self.last_metrics[evaluator_key][metric_key]:
                    if metric_key not in self.log_thresholds or \
                            (self.log_thresholds[metric_key][0]
                             <= stat
                             <= self.log_thresholds[metric_key][1]):
                        cur_eval_stats.append(f'{value}: {stat}')
                    self.last_metrics[evaluator_key][metric_key] = stat
                    mlflow.log_metric(mlflow.format_key(
                        f"{evaluator_key}_{metric_key}_{self.n_try}"), stat)
            if len(cur_eval_stats) > 0:
                logger.info(
                    f'Statistics for {evaluator_key.to_kv_str()}: {" ".join(cur_eval_stats)}')
