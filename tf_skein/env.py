# The implementation is loosely based on CondaCreator from the knit library,
# available at https://github.com/dask/knit under the 4-clause BSD license.

import json
import logging
import os
import shutil
import sys
import typing
import warnings
from subprocess import check_output, Popen, PIPE, CalledProcessError
from sys import version_info as v
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


class Env(typing.NamedTuple):
    """An isolated Python environment.

    Attributes
    ----------
    name : str
        A human-readable name of the environment.

    python : str
        Python version in the MAJOR.MINOR.MICRO format. Defaults to the
        version of ``sys.executable``.

    packages : list of str
        A list of packages to install in the environment. The packages
        are installed via pip, therefore all of the following forms
        are supported::

            SomeProject>=1,<2
            git+https://github.com/org/SomeProject
            http://SomeProject.org/archives/SomeProject-1.0.4.tar.gz
            path/to/SomeProject

        See `Installing Packages <https://packaging.python.org/tutorials \
        /installing-packages>`_ for more examples.
    """
    name: str
    python: str = f"{v.major}.{v.minor}.{v.micro}"
    packages: typing.List[str] = []

    def create(self, root: str = os.path.dirname(__file__)) -> str:
        """Create the environment.

        The environment is created via ``conda``. However, all the
        packages other than the Python interpreter are installed via
        pip to allow for more flexibility.

        Parameters
        ----------
        root : str
            A temporary directory for the created environments. The
            layout is not guaranteed to be stable across releases and
            should not be relied upon.

        Returns
        -------
        env_zip_path : str
            Path to the zipped environment.
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

            logger.info("Installing packages into " + self.name)
            _call([env_python_bin, "-m", "pip", "install"] + self.packages)

        env_zip_path = env_path + ".zip"
        if not os.path.exists(env_zip_path):
            path = shutil.make_archive(
                self.name,
                "zip",
                root_dir=env_path)

            try:
                os.rename(path, env_zip_path)
            except OSError:
                os.remove(path)  # Cleanup on failure.
        return env_zip_path


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
    logger.debug(out)
    logger.debug(err)
    if proc.returncode:
        raise CalledProcessError(proc.returncode, cmd)
