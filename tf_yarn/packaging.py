import logging
import sys
import os
import uuid
import cluster_pack

from typing import Dict, Collection, Tuple

_logger = logging.getLogger(__name__)


# this module has been moved to a new project
# moved to https://github.com/criteo/cluster-pack/blob/master/cluster_pack/packaging.py


def _get_tmp_dir():
    tmp_dir = f"/tmp/{uuid.uuid1()}"
    _logger.debug(f"local tmp_dir {tmp_dir}")
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def zip_path(py_dir: str, include_base_name=True, tmp_dir: str = _get_tmp_dir()):
    return cluster_pack.zip_path(py_dir, include_base_name, tmp_dir)


def get_editable_requirements(executable: str = sys.executable):
    return cluster_pack.get_editable_requirements(executable)


def get_non_editable_requirements(executable: str = sys.executable):
    return cluster_pack.get_non_editable_requirements(executable)


def upload_zip_to_hdfs(zip_file: str, archive_on_hdfs: str = None):
    return cluster_pack.upload_zip(zip_file, archive_on_hdfs)


def upload_env_to_hdfs(
        archive_on_hdfs: str = None,
        packer=None,
        additional_packages: Dict[str, str] = {},
        ignored_packages: Collection[str] = []
) -> Tuple[str, str]:
    return cluster_pack.upload_env(archive_on_hdfs, packer, additional_packages, ignored_packages)


def detect_packer_from_file(zip_file: str) -> cluster_pack.Packer:
    return cluster_pack.detect_packer_from_file(zip_file)


def get_editable_requirements_from_current_venv(
    executable: str = sys.executable,
    editable_packages_dir: str = os.getcwd()
):
    return cluster_pack.get_editable_requirements(executable, editable_packages_dir)


def get_default_fs():
    return cluster_pack.get_default_fs()
