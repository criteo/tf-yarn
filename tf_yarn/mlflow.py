import logging

from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import mlflow
    from mlflow.tracking import utils
except ModuleNotFoundError:
    pass

_use_mlflow: Optional[bool] = None


def use_mlflow() -> bool:
    global _use_mlflow
    if _use_mlflow is None:
        _use_mlflow = _detect_mlflow()
    return _use_mlflow


def _detect_mlflow() -> bool:
    try:
        import mlflow
        from mlflow.tracking import utils
    except ModuleNotFoundError:
        logger.warning("mlflow not installed")
        return False

    if not utils.is_tracking_uri_set():
        logger.warning("mlflow tracking uri is not set. will not use mlflow")
        return False

    return True


def active_run_id() -> str:
    if use_mlflow():
        active_run = mlflow.active_run()
        if not active_run:
            logger.warning("there is no active mlflow run. starting a new run ..")
            active_run = mlflow.start_run()
        return active_run.info.run_id
    else:
        return ""


def get_tracking_uri() -> str:
    if use_mlflow():
        return utils.get_tracking_uri()
    else:
        return ""


def set_tag(key: str, value: Any):
    if use_mlflow():
        mlflow.set_tag(key, value)


def set_tags(tags: Dict[str, Any]):
    if use_mlflow():
        mlflow.set_tags(tags)


def log_param(key: str, value: Any):
    if use_mlflow():
        mlflow.log_param(key, value)


def log_params(params: Dict[str, Any]):
    if use_mlflow():
        mlflow.log_params(params)


def log_metric(key: str, value: float, step: int = None):
    if use_mlflow():
        mlflow.log_metric(key, value, step)


def log_metrics(metrics: Dict[str, Any], step: int = None):
    if use_mlflow():
        mlflow.log_metrics(metrics, step)


def format_key(key: str) -> str:
    if key:
        return key.replace(":", "_").replace("/", "_")
    else:
        return ""
