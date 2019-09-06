import logging

from typing import Dict, Any

logger = logging.getLogger(__name__)

use_mlflow = False
try:
    import mlflow
    from mlflow.tracking import utils
    use_mlflow = True
except ModuleNotFoundError:
    pass


def active_run_id() -> str:
    if use_mlflow:
        if not utils.is_tracking_uri_set():
            logger.warning("mlflow tracking uri is not set. "
                           "mlflow will write outputs to current dir")

        active_run = mlflow.active_run()
        if not active_run:
            logger.warning("there is no active mlflow run. starting a new run ..")
            active_run = mlflow.start_run()
        return active_run.info.run_id
    else:
        return ""


def get_tracking_uri() -> str:
    return utils.get_tracking_uri()


def set_tag(key: str, value: Any):
    if use_mlflow:
        mlflow.set_tag(key, value)


def set_tags(tags: Dict[str, Any]):
    if use_mlflow:
        mlflow.set_tags(tags)


def log_param(key: str, value: Any):
    if use_mlflow:
        mlflow.log_param(key, value)


def log_params(params: Dict[str, Any]):
    if use_mlflow:
        mlflow.log_params(params)


def log_metric(key: str, value: float, step: int = None):
    if use_mlflow:
        mlflow.log_metric(key, value, step)


def log_metrics(metrics: Dict[str, Any], step: int = None):
    if use_mlflow:
        mlflow.log_metrics(metrics, step)


def format_key(key: str) -> str:
    if key:
        return key.replace(":", "_")
    else:
        return ""
