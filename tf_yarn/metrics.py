import logging
from typing import List, Dict
import skein


logger = logging.getLogger(__name__)


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
            ret = True
        return ret
