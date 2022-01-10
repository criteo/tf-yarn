import logging
from datetime import timedelta
from typing import (
    List,
    Optional,
    NamedTuple,
    Dict,
    Tuple
)

import skein

from tf_yarn import mlflow

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
