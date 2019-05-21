import getpass
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import typing
import uuid
import zipfile
import tensorflow as tf

try:
    import conda_pack
except NotImplementedError:
    # conda is not supported on windows
    pass
from urllib import parse
from pex.fetcher import PyPIFetcher
from pex.pex_builder import PEXBuilder
from pex.resolvable import Resolvable
from pex.resolver import resolve_multi, Unsatisfiable, Untranslateable
from pex.resolver_options import ResolverOptionsBuilder

from tf_yarn import _criteo

CRITEO_PYPI_URL = "http://build-nexus.prod.crto.in/repository/pypi/simple"

CONDA_DEFAULT_ENV = 'CONDA_DEFAULT_ENV'

_logger = logging.getLogger(__name__)


def zip_path(py_dir: str, include_base_name=True):
    """
    Zip current directory

    :param py_dir: directory to zip
    :param include_base_name: include the basename of py_dir into the archive (
        for skein zip files it should be False,
        for pyspark zip files it should be True)
    :return: destination of the archive
    """
    tmp_dir = _get_tmp_dir()
    py_archive = os.path.join(
        tmp_dir,
        os.path.basename(py_dir) + '.zip'
    )

    with zipfile.ZipFile(py_archive, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(py_dir):
            for file in files:
                # do not include .pyc files, it makes the import
                # fails for no obvious reason
                if file.endswith(".py"):
                    zipf.write(
                        os.path.join(root, file),
                        os.path.join(
                            os.path.basename(py_dir) if include_base_name else "",
                            os.path.relpath(root, py_dir),
                            file
                        )
                        if root != py_dir
                        else os.path.join(
                            os.path.basename(root) if include_base_name else "",
                            file
                        ))
    return py_archive


def pack_in_pex(requirements: typing.Dict[str, str], output: str
                ) -> str:
    """
    Pack current environment using a pex.

    :param requirements: list of requirements (ex ['tensorflow==0.1.10'])
    :param output: location of the pex
    :return: destination of the archive, name of the pex
    """
    requirements_to_install = [name + "==" + version
                               for name, version in requirements.items()]

    fetchers = []
    if _criteo.is_criteo():
        fetchers.append(PyPIFetcher(pypi_base=CRITEO_PYPI_URL))
    fetchers.append(PyPIFetcher())
    resolver_option_builder = ResolverOptionsBuilder(
        use_manylinux=True,
        fetchers=fetchers)
    resolvables = [Resolvable.get(req, resolver_option_builder) for
                   req in requirements_to_install]
    pex_builder = PEXBuilder(copy=True)

    try:
        resolveds = resolve_multi(resolvables, use_manylinux=True)
        for resolved in resolveds:
            _logger.debug("Add requirement %s", resolved.distribution)
            pex_builder.add_distribution(resolved.distribution)
            pex_builder.add_requirement(resolved.requirement)
    except (Unsatisfiable, Untranslateable):
        _logger.exception('Cannot create pex')
        raise

    pex_builder.build(output)

    return output


def _get_packages(editable: bool, executable: str = sys.executable):
    editable_mode = "-e" if editable else "--exclude-editable"
    results = subprocess.check_output(
        [f"{executable}", "-m", "pip", "list", "-l",
         f"{editable_mode}", "--format", "json"]).decode()

    _logger.debug(f"'pip list' with editable={editable} results:" + results)

    parsed_results = json.loads(results)

    # https://pip.pypa.io/en/stable/reference/pip_freeze/?highlight=freeze#cmdoption--all
    # freeze hardcodes to ignore those packages: wheel, distribute, pip, setuptools
    # To be iso with freeze we also remove those packages
    return [element for element in parsed_results
            if element["name"] not in
            ["distribute", "wheel", "pip", "setuptools"]]


def pack_current_venv_in_pex(output: str, reqs: typing.Dict[str, str]) -> str:
    """
    Pack current environment using a pex

    :param output: location of the pex
    :return: destination in hdfs, name of the pex
    """
    return pack_in_pex(reqs, output)


class Packer(typing.NamedTuple):
    env_name: str
    extension: str
    pack: typing.Callable[[str, typing.Dict[str, str]], str]


def get_env_name(env_var_name) -> str:
    """
    Return default virtual env
    """
    virtual_env_path = os.environ.get(env_var_name)
    if not virtual_env_path:
        return 'default'
    else:
        return os.path.basename(virtual_env_path)


CONDA_PACKER = Packer(
    get_env_name(CONDA_DEFAULT_ENV),
    'zip',
    lambda output, reqs: conda_pack.pack(output=output)
)
PEX_PACKER = Packer(
    get_env_name('VIRTUAL_ENV'),
    'pex',
    pack_current_venv_in_pex
)


def get_editable_requirements(executable: str = sys.executable):
    def _get(name):
        pkg = __import__(name.replace("-", "_"))
        return os.path.dirname(pkg.__file__)
    return [_get(package["name"]) for package in _get_packages(True, executable)]


def get_non_editable_requirements(executable: str = sys.executable):
    return _get_packages(False, executable)


def _get_archive_metadata_path(archive_on_hdfs: str) -> str:
    url = parse.urlparse(archive_on_hdfs)
    return url._replace(path=str(pathlib.Path(url.path).with_suffix('.json'))).geturl()


def _is_archive_up_to_date(archive_on_hdfs: str,
                           current_packages_list: typing.Dict[str, str]
                           ) -> bool:
    if not tf.gfile.Exists(archive_on_hdfs):
        return False
    archive_meta_data = _get_archive_metadata_path(archive_on_hdfs)
    if not tf.gfile.Exists(archive_meta_data):
        _logger.debug(f'metadata for archive {archive_on_hdfs} does not exist')
        return False
    with tf.gfile.GFile(archive_meta_data, "rb") as fd:
        packages_installed = json.loads(fd.read())
        return sorted(packages_installed.items()) == sorted(current_packages_list.items())


def _dump_archive_metadata(archive_on_hdfs: str,
                           current_packages_list: typing.Dict[str, str]
                           ):
    archive_meta_data = _get_archive_metadata_path(archive_on_hdfs)
    with tempfile.TemporaryDirectory() as tempdir:
        tempfile_path = os.path.join(tempdir, "metadata.json")
        with open(tempfile_path, "w") as fd:
            fd.write(json.dumps(current_packages_list, sort_keys=True, indent=4))
        if tf.gfile.Exists(archive_meta_data):
            tf.gfile.Remove(archive_meta_data)
        tf.gfile.Copy(tempfile_path, archive_meta_data)


def upload_env_to_hdfs(
        archive_on_hdfs: str = None,
        packer=None
) -> typing.Tuple[str, str]:
    if packer is None:
        if _is_conda_env():
            packer = CONDA_PACKER
        else:
            packer = PEX_PACKER

    pex_file = get_current_pex_filepath()
    env_name = os.path.basename(pex_file).split(
        '.')[0] if pex_file else packer.env_name
    if not archive_on_hdfs:
        archive_on_hdfs = (f"{get_default_fs()}/user/{getpass.getuser()}"
                           f"/envs/{env_name}.{packer.extension}")
    else:
        if pathlib.Path(archive_on_hdfs).suffix != f".{packer.extension}":
            raise ValueError(f"{archive_on_hdfs} has the wrong extension"
                             f", .{packer.extension} is expected")

    if not pex_file:
        upload_env_to_hdfs_from_venv(archive_on_hdfs, packer)
    else:
        tf.gfile.MakeDirs(os.path.dirname(archive_on_hdfs))
        tf.gfile.Copy(pex_file, archive_on_hdfs, overwrite=True)
        # Remove previous metadata
        archive_meta_data = _get_archive_metadata_path(archive_on_hdfs)
        if tf.gfile.Exists(archive_meta_data):
            tf.gfile.Remove(archive_meta_data)

    return (archive_on_hdfs,
            env_name)


def upload_env_to_hdfs_from_venv(
        archive_on_hdfs: str,
        packer=PEX_PACKER
):
    current_packages = {package["name"]: package["version"]
                        for package in get_non_editable_requirements()}

    if not _is_archive_up_to_date(archive_on_hdfs, current_packages):
        _logger.info(
            f"Zipping and uploading your env to {archive_on_hdfs}"
        )

        tmp_dir = _get_tmp_dir()
        archive_local = packer.pack(
            output=f"{tmp_dir}/{packer.env_name}.{packer.extension}",
            reqs=current_packages
        )
        tf.gfile.MakeDirs(os.path.dirname(archive_on_hdfs))
        tf.gfile.Copy(archive_local, archive_on_hdfs, overwrite=True)

        _dump_archive_metadata(archive_on_hdfs, current_packages)

        shutil.rmtree(tmp_dir)
    else:
        _logger.info(f"{archive_on_hdfs} already exists on hdfs")


def get_current_pex_filepath() -> typing.Optional[str]:
    """
    If we run from a pex, returns the path
    """
    pex_paths = [path for path in sys.path if path.endswith('.pex')]
    if pex_paths:
        return pex_paths[0]
    return None


def get_editable_requirements_from_current_venv():
    if get_current_pex_filepath():
        return dict()
    files = {}
    for requirement_dir in get_editable_requirements():
        files[os.path.basename(requirement_dir)] = requirement_dir
    return files


def get_default_fs():
    return subprocess.check_output("hdfs getconf -confKey fs.defaultFS".split()).strip().decode()


def _get_tmp_dir():
    tmp_dir = f"/tmp/{uuid.uuid1()}"
    _logger.debug(f"local tmp_dir {tmp_dir}")
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def _is_conda_env():
    return os.environ.get(CONDA_DEFAULT_ENV) is not None
