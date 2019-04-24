import glob
import logging
import os
import platform
import shutil
import socket
import tempfile
from typing import (
    Optional,
    Tuple,
    List,
    Iterable,
    Iterator
)
from contextlib import contextmanager
from subprocess import Popen, CalledProcessError, PIPE
from threading import Thread
from ._criteo import get_requirements_file

import setuptools

logger = logging.getLogger(__name__)

here = os.path.dirname(__file__)


class MonitoredThread(Thread):
    """A thread which captures any exception occurred during the
    execution of ``target``.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._exc = None

    @property
    def state(self):
        if self.is_alive():
            return "RUNNING"
        return "FAILED" if self.exception is not None else "SUCCEEDED"

    @property
    def exception(self) -> Optional[Exception]:
        return self._exc

    def run(self):
        try:
            super().run()
        except Exception as exc:
            self._exc = exc


def get_so_reuseport():
    try:
        return socket.SO_REUSEPORT
    except AttributeError:
        if platform.system() == "Linux":
            major, minor, *_ = platform.release().split(".")
            if (int(major), int(minor)) > (3, 9):
                # The interpreter must have been compiled on Linux <3.9.
                return 15
    return None


@contextmanager
def reserve_sock_addr() -> Iterator[Tuple[str, int]]:
    """Reserve an available TCP port to listen on.

    The reservation is done by binding a TCP socket to port 0 with
    ``SO_REUSEPORT`` flag set (requires Linux >=3.9). The socket is
    then kept open until the generator is closed.

    To reduce probability of 'hijacking' port, socket should stay open
    and should be closed _just before_ starting of ``tf.train.Server``
    """
    so_reuseport = get_so_reuseport()
    if so_reuseport is None:
        raise RuntimeError(
            "SO_REUSEPORT is not supported by the operating system") from None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, so_reuseport, 1)
        sock.bind(("", 0))
        _ipaddr, port = sock.getsockname()
        yield (socket.getfqdn(), port)


def iter_tasks(tasks: List[Tuple[str, int]]) -> Iterable[str]:
    """Iterate the tasks in a TensorFlow cluster.
    """
    for task_type, n_instances in tasks:
        yield from (f"{task_type}:{task_id}" for task_id in range(n_instances))


def xset_environ(**kwargs):
    """Exclusively set keys in the environment."""
    for key, value in kwargs.items():
        if key in os.environ:
            raise RuntimeError(f"{key} already set in os.environ: {value}")

    os.environ.update(kwargs)


def zip_path(path: str, tempdir: str):
    assert os.path.exists(path) and os.path.isdir(path)

    zip_path = os.path.join(tempdir, os.path.basename(path) + ".zip")
    created = shutil.make_archive(
        zip_path,
        "zip",
        root_dir=path)

    try:
        shutil.move(created, zip_path)
    except OSError as e:
        os.remove(created)  # Cleanup on failure.
        raise e from None
    return zip_path


class StaticDefaultDict(dict):
    """A ``dict`` with a static default value.

    Unlike ``collections.defaultdict`` this implementation does not
    implicitly update the mapping when queried with a missing key::

        >>> d = StaticDefaultDict(default=42)
        >>> d["foo"]
        42
        >>> d
        {}
    """
    def __init__(self, *args, default, **kwargs):
        super().__init__(*args, **kwargs)
        self.default = default

    def __missing__(self, key):
        return self.default


def create_and_pack_conda_env(
    name: str,
    python: str,
    pip_packages: List[str],
    root: Optional[str] = tempfile.tempdir
) -> str:
    """Create a Conda environment.

    The environment is created via ``conda``. However, all the
    packages other than the Python interpreter are installed via
    ``pip`` to allow for more flexibility.

    Parameters
    ----------
    name : str
        A human-readable name of the environment.

    python : str
        Python version in the MAJOR.MINOR.MICRO format.

    pip_packages : list
        PyPI packages to install in the environment.

    root : list
        Path to the root directory with Conda environments. If ``None``,
        system temporary directory will be used with a fallback to the
        current directory.

    Returns
    -------
    env_path : str
        Path to the packed environment.
    """
    try:
        _call(["conda"])
    except CalledProcessError:
        raise RuntimeError("conda is not available in $PATH")

    env_path = os.path.join(root or os.getcwd(), name)
    if not os.path.exists(env_path):
        logger.info("Creating new env " + name)
        _call([
            "conda", "create", "-p", env_path, "-y", "-q", "--copy",
            "python=" + python,
            # TensorFlow enforces an upper bound on setuptools which
            # conflicts with the version installed by Conda by default.
            "setuptools=" + setuptools.__version__
        ], env=dict(os.environ))

        env_python_bin = os.path.join(env_path, "bin", "python")
        if not os.path.exists(env_python_bin):
            raise RuntimeError(
                "Failed to create Python binary at " + env_python_bin)

        if pip_packages:
            logger.info("Installing packages into " + name)
            _call([env_python_bin, "-m", "pip", "install"] +
                  pip_packages)

            requirements_path = os.path.join(env_path, get_requirements_file())
            with open(requirements_path, "w") as f:
                print(*pip_packages, sep=os.linesep, file=f)

    env_zip_path = env_path + ".zip"
    if not os.path.exists(env_zip_path):
        import conda_pack
        conda_pack.pack(prefix=env_path, output=env_zip_path)
    return env_zip_path


def _call(cmd, **kwargs):
    logger.info(" ".join(cmd))
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, **kwargs)
    out, err = proc.communicate()
    if proc.returncode:
        logger.error(out)
        logger.error(err)
        raise CalledProcessError(proc.returncode, cmd)
    else:
        logger.debug(out)
        logger.debug(err)
