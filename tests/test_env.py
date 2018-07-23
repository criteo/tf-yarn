import os
from subprocess import check_call

from tf_skein import Env


def test_env_create(tmpdir):
    env = Env(
        name="test",
        packages=["pycodestyle"])
    env_path = env.create(str(tmpdir))

    env_python_bin = os.path.join(env_path, "bin", "python")
    check_call([env_python_bin, "-m", "pycodestyle", "--version"])
