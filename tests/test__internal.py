import errno
import os
import socket
import subprocess
import sys
import tarfile
import zipfile
from subprocess import check_output
from zipfile import ZipFile, is_zipfile

import pytest

from tf_yarn import TaskSpec, generate_services_using_pex
from tf_yarn._internal import (
    MonitoredThread,
    reserve_sock_addr,
    dump_fn,
    load_fn,
    xset_environ,
    zip_inplace,
    StaticDefaultDict,
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


def test_encode_fn_decode_fn(tmpdir):
    def g(x):
        return x

    def f():
        return g(42)

    path = tmpdir.join("f.dill")
    dump_fn(f, path)
    assert load_fn(path)() == f()


def test_zip_inplace(tmpdir):
    s = "Hello, world!"
    tmpdir.mkdir("foo").join("bar.txt").write_text(s, encoding="utf-8")
    b = 0xffff.to_bytes(4, "little")
    tmpdir.join("boo.bin").write_binary(b)

    zip_path = zip_inplace(str(tmpdir))
    assert os.path.isfile(zip_path)
    assert zip_path.endswith(".zip")
    assert is_zipfile(zip_path)
    with ZipFile(zip_path) as zf:
        zipped = {zi.filename for zi in zf.filelist}
        assert "foo/" in zipped
        assert "foo/bar.txt" in zipped
        assert "boo.bin" in zipped

        assert zf.read("foo/bar.txt") == s.encode()
        assert zf.read("boo.bin") == b


def test_zip_inplace_replace(tmpdir):
    zip_path = zip_inplace(str(tmpdir))
    os.truncate(zip_path, 0)
    assert os.path.getsize(zip_path) == 0
    zip_inplace(str(tmpdir), replace=True)
    assert os.path.getsize(zip_path) > 0


def test_static_default_dict():
    d = StaticDefaultDict({"foo": 42}, default=100500)
    assert d["foo"] == 42
    assert d["bar"] == 100500
    assert "bar" not in d


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


def test_generate_services_pex():
    pex_filename="mypex.pex"
    pex_folder='/some/where/'
    env = {"VAR0": "VALUE0", "VAR1": "VALUE1"}
    files = {"hello.py": "file:///tmp/hello.py"}
    task_name = "task0"
    task_specs = {task_name: TaskSpec(memory=2, vcores=3),
                  'ps': TaskSpec(memory=1, vcores=2),
                  'worker': TaskSpec(memory=1, vcores=1)}
    services = generate_services_using_pex(pex_folder + pex_filename, env, files, task_specs)

    assert len(services) == 3, f"services size is {len(services)}. Expected 3"
    assert task_name in services

    service = services[task_name]
    # Env variable are forwarded
    assert all(item in service.env.items() for item in env.items()), f"{env.items()} must be included in {service.env.items()}"
    # Pex env variable are set
    assert service.env["PEX_ROOT"].startswith('/tmp/.pex')
    assert service.env["PEX_MODULE"] == "tf_yarn._dispatch_task"
    # it calls pex
    assert service.commands[0].startswith('./'+pex_filename)
    # pex is uploaded with files
    files_in_services = {key: value.source for key, value in service.files.items()}
    assert all(item in files_in_services.items() for item in files.items()), f"{files.items()} must be included in {files_in_services.items()}"
    assert files_in_services[pex_filename] == 'file://' + pex_folder + pex_filename
