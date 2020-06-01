import os
from typing import (
    Optional,
    NamedTuple
)

import cluster_pack
from tf_yarn import topologies


class PythonEnvDescription(NamedTuple):
    path_to_archive: str
    dispatch_task_cmd: str
    dest_path: str


INDEPENDENT_WORKERS_MODULE = "tf_yarn.tasks._independent_workers_task"
TENSORBOARD_MODULE = "tf_yarn.tasks._tensorboard_task"
CONDA_ENV_NAME = "pyenv"
CONDA_CMD = f"{CONDA_ENV_NAME}/bin/python"


def gen_pyenv_from_existing_archive(path_to_archive: str) -> PythonEnvDescription:

    archive_filename = os.path.basename(path_to_archive)

    packer = cluster_pack.detect_packer_from_file(path_to_archive)
    if packer == cluster_pack.PEX_PACKER:
        return PythonEnvDescription(
            path_to_archive,
            f"./{archive_filename}",
            archive_filename)
    elif packer == cluster_pack.CONDA_PACKER:
        return PythonEnvDescription(
            path_to_archive,
            f"{CONDA_CMD}", CONDA_ENV_NAME)
    else:
        raise ValueError("Archive format unsupported. Must be .pex or conda .zip")


def gen_task_cmd(pyenv: PythonEnvDescription,
                 task_type: str,
                 custom_task_module: Optional[str]) -> str:

    if task_type == "tensorboard":
        containers_module = TENSORBOARD_MODULE
    elif task_type in topologies.ALL_TASK_TYPES:
        if custom_task_module:
            containers_module = custom_task_module
        else:
            containers_module = INDEPENDENT_WORKERS_MODULE
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    return f"{pyenv.dispatch_task_cmd} -m {containers_module} "
