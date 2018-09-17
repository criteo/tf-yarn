# Copyright 2018 Criteo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import logging
import os
import platform
import shutil
import socket
import tempfile
import typing
from contextlib import contextmanager
from subprocess import Popen, CalledProcessError, PIPE
from threading import Thread

import dill
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

    The acquired TCP socket is created with ``SO_REUSEPORT`` flag set
    and is kept open until the generator is closed.
    """
    try:
        so_reuseport = socket.SO_REUSEPORT
    except AttributeError:
        if platform.system() == "Linux" and platform.release() >= "3.9":
            # The interpreter must have been compiled on Linux <3.9.
            so_reuseport = 15
        else:
            raise RuntimeError(
                "SO_REUSEPORT is not supported by the operating system")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, so_reuseport, 1)
        sock.bind(("", 0))
        _ipaddr, port = sock.getsockname()
        yield (socket.gethostname(), port)


def iter_tasks(num_workers, num_ps) -> typing.Iterable[str]:
    """Iterate the tasks in a TensorFlow cluster.

    Note that ``"evaluator"`` is not part of the cluster.
    """
    yield "chief:0"
    yield from (f"worker:{task_id}" for task_id in range(num_workers))
    yield from (f"ps:{task_id}" for task_id in range(num_ps))


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


def expand_wildcards_in_classpath(classpath: str) -> str:
    """Expand wildcard entries in the $CLASSPATH.

    JNI-invoked JVM does not support wildcards in the classpath. This
    function replaces all classpath entries of the form ``foo/bar/*``
    with the JARs in the ``foo/bar`` directory.

    See "Common Problems" section in the libhdfs docs
    https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-hdfs/LibHdfs.html
    """
    def maybe_glob(path):
        if path.endswith("*"):
            yield from glob.iglob(path + ".jar")
        else:
            yield path

    return ":".join(entry for path in classpath.split(":")
                    for entry in maybe_glob(path))


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
    pip_packages: typing.List[str],
    root: typing.Optional[str] = tempfile.tempdir
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

            requirements_path = os.path.join(env_path, "requirements.txt")
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
