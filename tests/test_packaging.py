import contextlib
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock

import pytest

from tf_yarn import packaging

MODULE_TO_TEST = "tf_yarn.packaging"
PYTHON_SYSTEM_EXEC = '/bin/python3.6'
SKIP_REASON = f'system python must be installed at {PYTHON_SYSTEM_EXEC}'
MYARCHIVE_FILENAME = "myarchive.pex"
MYARCHIVE_METADATA = "myarchive.json"
VARNAME = 'VARNAME'


def test_get_virtualenv_name():
    with mock.patch.dict('os.environ'):
        os.environ[VARNAME] = '/path/to/my_venv'
        assert 'my_venv' == packaging.get_env_name(VARNAME)


def test_get_virtualenv_empty_returns_default():
    with mock.patch.dict('os.environ'):
        if VARNAME in os.environ:
            del os.environ[VARNAME]
        assert 'default' == packaging.get_env_name(VARNAME)


@pytest.mark.skipif(not os.path.exists(PYTHON_SYSTEM_EXEC),
                    reason=SKIP_REASON)
def test_get_empty_editable_requirements():
    with tempfile.TemporaryDirectory() as tempdir:
        _create_venv(tempdir)
        subprocess.check_call([
                        f"{tempdir}/bin/python", "-m", "pip", "install",
                        "cloudpickle", _get_editable_package_name(), "pip==18.1"
                        ])
        editable_requirements = packaging.get_editable_requirements(f"{tempdir}/bin/python")
        assert len(editable_requirements) == 0


@pytest.mark.skipif(not os.path.exists(PYTHON_SYSTEM_EXEC),
                    reason=SKIP_REASON)
def test_get_empty_non_editable_requirements():
    with tempfile.TemporaryDirectory() as tempdir:
        _create_venv(tempdir)
        subprocess.check_call([
                    f"{tempdir}/bin/python", "-m", "pip", "install",
                    "-e", _get_editable_package_name(), "pip==18.1"
                    ])
        non_editable_requirements = packaging.get_non_editable_requirements(f"{tempdir}/bin/python")
        assert len(non_editable_requirements) == 0


@pytest.mark.skipif(not os.path.exists(PYTHON_SYSTEM_EXEC),
                    reason=SKIP_REASON)
def test_get_editable_requirements():
    with tempfile.TemporaryDirectory() as tempdir:
        _create_venv(tempdir)
        _pip_install(tempdir)
        editable_requirements = packaging.get_editable_requirements(f"{tempdir}/bin/python")
        assert len(editable_requirements) == 1
        assert os.path.basename(editable_requirements[0]) == "user_lib"


@pytest.mark.skipif(not os.path.exists(PYTHON_SYSTEM_EXEC),
                    reason=SKIP_REASON)
def test_get_non_editable_requirements():
    with tempfile.TemporaryDirectory() as tempdir:
        print("tempdir " + tempdir)
        _create_venv(tempdir)
        _pip_install(tempdir)
        non_editable_requirements = packaging.get_non_editable_requirements(f"{tempdir}/bin/python")
        print(f"non_editable_packages: {non_editable_requirements}")
        assert len(non_editable_requirements) == 1
        assert non_editable_requirements[0]["name"] == "cloudpickle"


def _create_venv(tempdir: str):
    subprocess.check_call([PYTHON_SYSTEM_EXEC, "-m", "venv", f"{tempdir}"])


def _pip_install(tempdir: str):
    subprocess.check_call([f"{tempdir}/bin/python", "-m", "pip", "install",
                           "cloudpickle", "pip==18.1"])
    pkg = _get_editable_package_name()
    print("pgk=" + pkg)
    subprocess.check_call([f"{tempdir}/bin/python", "-m", "pip", "install", "-e", pkg])
    if pkg not in sys.path:
        sys.path.append(pkg)


def _get_editable_package_name():
    return os.path.join(os.path.dirname(__file__), "user-lib")


@mock.patch(f"{MODULE_TO_TEST}.tf")
def test_update_no_archive(mock_tf):
    map_is_exist = {MYARCHIVE_FILENAME: False}
    mock_tf.gfile.Exists.side_effect = lambda arg: map_is_exist[arg]
    assert not packaging._is_archive_up_to_date(MYARCHIVE_FILENAME, [])


@mock.patch(f"{MODULE_TO_TEST}.tf")
def test_update_no_metadata(mock_tf):
    map_is_exist = {MYARCHIVE_FILENAME: True,
                    MYARCHIVE_METADATA: False}
    mock_tf.gfile.Exists.side_effect = lambda arg: map_is_exist[arg]
    assert not packaging._is_archive_up_to_date(MYARCHIVE_FILENAME, [])


@pytest.mark.parametrize("current_packages, metadata_packages, expected", [
    pytest.param({"a": "2.0", "b": "1.0"}, {"a": "2.0", "b": "1.0"}, True),
    pytest.param({"a": "2.0", "b": "1.0"}, {"a": "1.0", "b": "1.0"}, False),
    pytest.param({"a": "2.0", "b": "1.0"}, {"a": "2.0"}, False),
    pytest.param({"a": "2.0"}, {"a": "2.0", "b": "1.0"}, False),
    pytest.param({}, {"a": "2.0", "b": "1.0"}, False),
    pytest.param({"a": "2.0"}, {"c": "1.0"}, False),
    pytest.param({}, {}, True),
])
def test_update_version_comparaison(current_packages, metadata_packages,
                                    expected):
    map_is_exist = {MYARCHIVE_FILENAME: True,
                    MYARCHIVE_METADATA: True}
    with mock.patch(f"{MODULE_TO_TEST}.tf") as mock_tf:
        mock_tf.gfile.Exists.side_effect = map_is_exist
        # Mock metadata on hdfs
        gFile = mock.MagicMock()
        gFile.read.return_value = json.dumps(metadata_packages)
        gFile.__enter__.return_value = gFile
        mock_tf.gfile.GFile.return_value = gFile
        # Test if package is updated
        assert packaging._is_archive_up_to_date(MYARCHIVE_FILENAME,
                                                current_packages) == expected


def Any(cls):
    class Any(cls):
        def __eq__(self, other):
            return isinstance(other, cls)
    return Any()


expected_file = """\
{
    "a": "1.0",
    "b": "2.0"
}"""


@mock.patch(f"{MODULE_TO_TEST}.tf")
def test_dump_metadata(mock_tf):
    mock_open = mock.mock_open()
    with mock.patch(f"{MODULE_TO_TEST}.open", mock_open):
        mock_tf.gfile.Exists.return_value = True
        packages = {"a": "1.0", "b": "2.0"}
        packaging._dump_archive_metadata(MYARCHIVE_FILENAME, packages)
        # Check previous file has been deleted
        mock_tf.gfile.Remove.assert_called_once_with(MYARCHIVE_METADATA)
        # Check file is ok
        mock_fd = mock_open()
        mock_fd.write.assert_called_once_with(expected_file)
        mock_tf.gfile.Copy.assert_called_once_with(Any(str), MYARCHIVE_METADATA)


def test_upload_env():
    mock_packer = mock.MagicMock(spec=packaging.Packer)
    with contextlib.ExitStack() as stack:
        # Mock all objects
        mock_is_archive = stack.enter_context(
                mock.patch(f"{MODULE_TO_TEST}._is_archive_up_to_date"))
        mock_get_packages = stack.enter_context(
                mock.patch(f"{MODULE_TO_TEST}.get_non_editable_requirements"))
        mock_tf = stack.enter_context(mock.patch(f"{MODULE_TO_TEST}.tf"))
        stack.enter_context(mock.patch(f"{MODULE_TO_TEST}._dump_archive_metadata"))
        stack.enter_context(mock.patch(f"{MODULE_TO_TEST}.shutil.rmtree"))

        # Regenerate archive
        mock_is_archive.return_value = False
        mock_get_packages.return_value = [{"name": "a", "version": "1.0"},
                                          {"name": "b", "version": "2.0"}]
        mock_packer.pack.return_value = MYARCHIVE_FILENAME
        mock_packer.extension = "pex"
        packaging.upload_env_to_hdfs(MYARCHIVE_FILENAME, mock_packer)
        mock_packer.pack.assert_called_once_with(output=Any(str), reqs={"a": "1.0",
                                                                        "b": "2.0"})
        mock_tf.gfile.Copy.assert_called_once_with(MYARCHIVE_FILENAME,
                                                   MYARCHIVE_FILENAME,
                                                   overwrite=True)


def test_upload_env_to_hdfs_should_throw_error_if_wrong_extension():
    with pytest.raises(ValueError):
        packaging.upload_env_to_hdfs("myarchive.tar.gz", packer=packaging.CONDA_PACKER)


def test_upload_env_to_hdfs_in_a_pex():
    home_path = '/home/j.doe'
    home_hdfs_path = '/user/j.doe'
    with contextlib.ExitStack() as stack:
        mock_pex_filepath = stack.enter_context(
            mock.patch(f"{MODULE_TO_TEST}.get_current_pex_filepath"))
        mock_tf = stack.enter_context(mock.patch(f"{MODULE_TO_TEST}.tf"))
        mock__get_archive_metadata_path = stack.enter_context(
            mock.patch(f"{MODULE_TO_TEST}._get_archive_metadata_path")
        )
        mock__get_archive_metadata_path.return_value = f"{home_hdfs_path}/blah.json"
        mock_pex_filepath.return_value = f"{home_path}/myapp.pex"
        # Metadata already exists on hdfs
        mock_tf.gfile.Exists.return_value = True

        result = packaging.upload_env_to_hdfs(f'{home_hdfs_path}/blah.pex')

        mock_tf.gfile.MakeDirs.assert_called_once_with(home_hdfs_path)
        mock_tf.gfile.Copy.assert_called_once_with(
            f'{home_path}/myapp.pex', f'{home_hdfs_path}/blah.pex', overwrite=True)
        # Check metadata has been cleaned
        mock_tf.gfile.Remove.assert_called_once_with(f'{home_hdfs_path}/blah.json')
        # check envname
        assert 'myapp' == result[1]
