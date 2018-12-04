import pytest

from tf_yarn._env import (
    gen_pyenv_from_existing_archive,
    CONDA_CMD, CONDA_ENV_NAME,
    INDEPENDENT_WORKERS_MODULE,
    STANDALONE_CLIENT_MODULE
)

test_data = [
    ("/path/to/myenv.pex",
     f"./myenv.pex -m {INDEPENDENT_WORKERS_MODULE} ",
     "myenv.pex",
     False),
    ("/path/to/myenv.zip",
     f"{CONDA_CMD} -m {INDEPENDENT_WORKERS_MODULE}",
     CONDA_ENV_NAME,
     False),
    ("/path/to/myenv.pex",
     f"./myenv.pex -m {STANDALONE_CLIENT_MODULE} ",
     "myenv.pex",
     True),
    ("/path/to/myenv.zip",
     f"{CONDA_CMD} -m {STANDALONE_CLIENT_MODULE}",
     CONDA_ENV_NAME,
     True)
]


@pytest.mark.parametrize(
    "path_to_archive,expected_cmd, expected_dest_path, standalone_client_mode",
    test_data)
def test_gen_pyenvs_from_existing_env(path_to_archive, expected_cmd,
                                      expected_dest_path, standalone_client_mode):
    result = gen_pyenv_from_existing_archive(path_to_archive, standalone_client_mode)
    assert result.path_to_archive == path_to_archive
    assert result.dispatch_task_cmd == expected_cmd
    assert result.dest_path == expected_dest_path


def test_gen_pyenvs_from_unknown_format():
    with pytest.raises(ValueError):
        gen_pyenv_from_existing_archive("/path/to/pack.tar.bz2", standalone_client_mode=False)
