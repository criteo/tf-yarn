import json
import logging
import os
import shutil
import socket
import sys
import typing
import warnings
from contextlib import contextmanager
from subprocess import Popen, CalledProcessError, PIPE, check_output
from threading import Thread
from urllib.request import urlretrieve

import dill

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
    def exception(self) -> typing.Optional[Exception]:
        return self._exc

    def run(self):
        try:
            super().run()
        except Exception as exc:
            self._exc = exc


@contextmanager
def reserve_sock_addr() -> typing.Iterator[typing.Tuple[str, int]]:
    """Reserve an available TCP port to listen on.

    The acquired TCP socket is hold open until the generator is
    closed. This does not eliminate the chance of collision between
    multiple concurrent Python processes, but it makes it slightly
    less likely.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        _ipaddr, port = sock.getsockname()
        yield (socket.gethostname(), port)


def dump_fn(fn, path: str) -> None:
    """Dump a function to a file in an unspecified binary format."""
    with open(path, "wb") as file:
        dill.dump(fn, file, recurse=True)


def load_fn(path: str):
    """Load a function from a file produced by ``encode_fn``."""
    with open(path, "rb") as file:
        return dill.load(file)


def xset_environ(**kwargs):
    """Exclusively set keys in the environment."""
    for key, value in kwargs.items():
        if key in os.environ:
            raise RuntimeError(f"{key} already set in os.environ: {value}")

    os.environ.update(kwargs)


def zip_inplace(path, replace=False):
    assert os.path.exists(path) and os.path.isdir(path)

    zip_path = path + ".zip"
    if not os.path.exists(zip_path) or replace:
        created = shutil.make_archive(
            os.path.basename(path),
            "zip",
            root_dir=path)

        try:
            shutil.move(created, zip_path)
        except OSError as e:
            os.remove(created)  # Cleanup on failure.
            raise e from None
    return zip_path


def aggregate_from_kv(
    kv,
    stage: str,
    num_workers: int,
    num_ps: int
) -> typing.Dict[str, list]:
    """Aggregate values over a TensorFlow cluster for a given stage."""

    def get(target):
        return kv.wait(stage + "/" + target).decode()

    spec = {
        "chief": [get("chief_0")]
    }

    for idx in range(num_ps):
        spec.setdefault("ps", []).append(get(f"ps_{idx}"))

    for idx in range(num_workers):
        spec.setdefault("worker", []).append(get(f"worker_{idx}"))

    return spec


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


class PyEnv(typing.NamedTuple):
    """A Python environment.

    Attributes
    ----------
    name : str
        A human-readable name of the environment.

    python : str
        Python version in the MAJOR.MINOR.MICRO format.

    pip_packages : list
        Python packages to install in the environment.
    """
    name: str
    python: str
    pip_packages: typing.List[str]

    def create(self, root: str = here) -> str:
        """
        The environment is created via ``conda``. However, all the
        packages other than the Python interpreter are installed via
        pip to allow for more flexibility.

        Parameters
        ----------
        root : str, optional
            Root directory for the created environments. The layout
            is not guaranteed to be stable across releases and should
            not be relied upon.

        Returns
        -------
        env_path : str
            Path to the environment root.
        """
        try:
            conda_info = json.loads(
                check_output("conda info --json".split()).decode())
            conda_root = conda_info["conda_prefix"]
        except (OSError, IOError):
            warnings.warn("No conda found in PATH")
            conda_root = os.path.join(root, "tmp_conda")

        conda_bin = os.path.join(conda_root, "bin", "conda")
        if not os.path.exists(conda_bin):
            _install_miniconda(conda_root)
        conda_envs = os.path.join(conda_root, "envs")
        env_path = os.path.join(conda_envs, self.name)
        if not os.path.exists(env_path):
            logger.info("Creating new env " + self.name)
            _call([
                conda_bin, "create", "-p", env_path, "-y", "-q", "--copy",
                "python=" + self.python
            ], env=dict(os.environ))

            env_python_bin = os.path.join(env_path, "bin", "python")
            if not os.path.exists(env_python_bin):
                raise RuntimeError(
                    "Failed to create Python binary at " + env_python_bin)

            if self.pip_packages:
                logger.info("Installing packages into " + self.name)
                _call([env_python_bin, "-m", "pip", "install"] +
                      self.pip_packages)

                requirements_path = os.path.join(env_path, "requirements.txt")
                with open(requirements_path, "w") as f:
                    print(*self.pip_packages, sep=os.linesep, file=f)

        return env_path


def _install_miniconda(root: str):
    if os.path.exists(root):
        os.rmdir(root)  # Fail if non-empty.

    logger.debug("Downloading latest Miniconda.sh")
    installer_path, _ = urlretrieve(_miniconda_url())
    logger.debug("Installing Miniconda in " + root)
    _call(["bash", installer_path, "-b", "-p", root])


def _miniconda_url():
    if sys.platform.startswith("linux"):
        platform = "Linux"
    elif sys.platform.startswith("darwin"):
        platform = "MacOSX"
    else:
        raise RuntimeError(sys.platform + " is not supported")
    arch = "x86_64" if sys.maxsize > 2 ** 32 else "x86"
    return ("https://repo.continuum.io/miniconda/"
            f"Miniconda3-latest-{platform}-{arch}.sh")


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
