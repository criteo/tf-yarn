import errno
import os
import socket
import subprocess
import sys
import tempfile
import zipfile
from subprocess import check_output

import pytest

from tf_yarn._internal import (
    MonitoredThread,
    reserve_sock_addr,
    xset_environ,
    create_and_pack_conda_env,
)


def test_monitored_thread():
    def fail():
        raise RuntimeError(42)

    thread = MonitoredThread(target=fail)
    thread.start()
    thread.join()

    assert isinstance(thread.exception, RuntimeError)
    assert thread.exception.args == (42, )


def test_reserve_sock_addr():
    with reserve_sock_addr() as (host, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(OSError) as exc_info:
            sock.bind((host, port))

        # Ensure that the iterator holds the sockets open.
        assert exc_info.value.errno == errno.EADDRINUSE


def test_xset_environ(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    xset_environ(foo="boo")
    assert os.environ["foo"] == "boo"


def test_xset_environ_failure(monkeypatch):
    monkeypatch.setattr(os, "environ", {"foo": "bar"})
    with pytest.raises(RuntimeError):
        xset_environ(foo="boo")

    assert os.environ["foo"] == "bar"


def conda_is_available():
    p = subprocess.run(
        ["conda"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True)
    return p.returncode == 0


@pytest.mark.skipif(not conda_is_available(), reason="conda is not available")
def test_create_conda_env(tmpdir):
    env_zip_path = create_and_pack_conda_env(
        name="test",
        python="{0.major}.{0.minor}".format(sys.version_info),
        pip_packages=["pycodestyle"],
        root=str(tmpdir))
    assert os.path.isfile(env_zip_path)
    env_path, _zip = os.path.splitext(env_zip_path)
    assert os.path.isdir(env_path)

    env_unzipped_path = tmpdir.join("unzipped")
    with zipfile.ZipFile(env_zip_path) as zf:
        zf.extractall(env_unzipped_path)

    env_python_bin = os.path.join(env_unzipped_path, "bin", "python")
    os.chmod(env_python_bin, 0o755)
    check_output([env_python_bin, "-m", "pycodestyle", "--version"])
