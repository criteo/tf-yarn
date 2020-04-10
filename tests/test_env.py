import pytest

from tf_yarn._env import (
    gen_pyenv_from_existing_archive,
    CONDA_CMD, CONDA_ENV_NAME
)

test_data = [
    ("/path/to/myenv.pex",
     "./myenv.pex",
     "myenv.pex"),
    ("/path/to/myenv.zip",
     f"{CONDA_CMD}",
     CONDA_ENV_NAME)
]


@pytest.mark.parametrize(
    "path_to_archive,expected_cmd, expected_dest_path",
    test_data)
def test_gen_pyenvs_from_existing_env(path_to_archive, expected_cmd,
                                      expected_dest_path):
    result = gen_pyenv_from_existing_archive(path_to_archive)
    assert result.path_to_archive == path_to_archive
    assert result.dispatch_task_cmd == expected_cmd
    assert result.dest_path == expected_dest_path


def test_gen_pyenvs_from_unknown_format():
    with pytest.raises(ValueError):
        gen_pyenv_from_existing_archive("/path/to/pack.tar.bz2")
