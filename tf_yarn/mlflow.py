import logging
import tempfile
import os

from typing import Dict, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)

try:
    import mlflow
    from mlflow import exceptions
    # pulled by mlflow
    from requests import exceptions as requests_exc
except (ModuleNotFoundError, ImportError):
    pass

_use_mlflow: Optional[bool] = None


def use_mlflow() -> bool:
    global _use_mlflow
    if _use_mlflow is None:
        _use_mlflow = _detect_mlflow()
    return _use_mlflow


def _detect_mlflow() -> bool:
    tf_yarn_use_mlflow = os.getenv("TF_YARN_USE_MLFLOW", "")
    if tf_yarn_use_mlflow == "False":
        return False

    try:
        import mlflow
        from mlflow import exceptions
        from mlflow import tracking
        # pulled by mlflow
        from requests import exceptions as requests_exc
    except (ModuleNotFoundError, ImportError):
        logger.exception("mlflow is not installed")
        return False

    if not tracking.is_tracking_uri_set():
        logger.warning("mlflow tracking uri is not set. will not use mlflow")
        return False

    return True


def _is_pyarrow_installed():
    try:
        import pyarrow
        return True
    except ModuleNotFoundError:
        return False


def optional_mlflow(return_default=None):
    def optional_mlflow_decorator(f):
        @wraps(f)
        def wrapper(*args, **kwds):
            if use_mlflow():
                try:
                    return f(*args, **kwds)
                except (ConnectionError, requests_exc.RequestException, exceptions.MlflowException):
                    logger.exception("mlflow connection error")
            else:
                return return_default
        return wrapper
    return optional_mlflow_decorator


@optional_mlflow(return_default="")
def active_run_id() -> str:
    active_run = mlflow.active_run()
    if not active_run:
        logger.warning("there is no active mlflow run. starting a new run ..")
        active_run = mlflow.start_run()
    return active_run.info.run_id


@optional_mlflow(return_default="")
def get_tracking_uri() -> str:
    return mlflow.get_tracking_uri()


@optional_mlflow()
def set_tag(key: str, value: Any):
    mlflow.set_tag(key, value)


@optional_mlflow()
def set_tags(tags: Dict[str, Any]):
    mlflow.set_tags(tags)


@optional_mlflow()
def log_param(key: str, value: Any):
    mlflow.log_param(key, value)


@optional_mlflow()
def log_params(params: Dict[str, Any]):
    mlflow.log_params(params)


@optional_mlflow()
def log_metric(key: str, value: float, step: int = None):
    mlflow.log_metric(key, value, step)


@optional_mlflow()
def log_metrics(metrics: Dict[str, Any], step: int = None):
    mlflow.log_metrics(metrics, step)


@optional_mlflow()
def log_artifact(local_path: str, artifact_path: str = None):
    mlflow.log_artifact(local_path, artifact_path)


@optional_mlflow()
def log_artifacts(local_dir: str, artifact_path: str = None):
    mlflow.log_artifacts(local_dir, artifact_path)


def format_key(key: str) -> str:
    if key:
        return key.replace(":", "_").replace("/", "_")
    else:
        return ""


@optional_mlflow()
def save_text_to_mlflow(content, filename):
    if not _is_pyarrow_installed():
        logger.warning(f"Pyarrow is not installed. {filename} artifact won't be stored on HDFS")
        return

    logger.info(f"save file {filename} to mlflow")
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, filename)
        with open(path, 'wb') as f:
            f.write(content.encode())
        mlflow.log_artifact(path)
