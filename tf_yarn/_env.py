import typing
import os


class PythonEnvDescription(typing.NamedTuple):
    path_to_archive: str
    dispatch_task_cmd: str
    dest_path: str


DISPATCH_MODULE = "-m tf_yarn._dispatch_task"
CONDA_ENV_NAME = "pyenv"
CONDA_CMD = f"{CONDA_ENV_NAME}/bin/python {DISPATCH_MODULE}"


def gen_pyenv_from_existing_archive(path_to_archive: str
                                    ) -> PythonEnvDescription:
    archive_filename = os.path.basename(path_to_archive)
    if path_to_archive.endswith('.pex'):
        return PythonEnvDescription(
            path_to_archive,
            f"./{archive_filename} {DISPATCH_MODULE} ",
            archive_filename)
    elif path_to_archive.endswith(".zip"):
        return PythonEnvDescription(path_to_archive, CONDA_CMD, CONDA_ENV_NAME)
    else:
        raise ValueError("Archive format unsupported. Must be .pex or conda .zip")


def gen_task_cmd(pyenv: PythonEnvDescription,
                 num_ps: int,
                 num_workers: int,
                 log_conf_file: typing.Optional[str]) -> str:

    conf_args = f"--log-conf-file={log_conf_file}" if log_conf_file is not None else ""

    return (f"{pyenv.dispatch_task_cmd} "
            f"--num-ps={num_ps} --num-workers={num_workers} " + conf_args
            )
