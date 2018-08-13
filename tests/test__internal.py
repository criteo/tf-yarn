import errno
import os
import socket
import sys
from subprocess import check_output
from zipfile import ZipFile, is_zipfile

import pytest

from tf_skein._internal import (
    reserve_sock_addr,
    encode_fn,
    decode_fn,
    xset_environ,
    zip_inplace,
    StaticDefaultDict,
    PyEnv,
)


def test_iter_available_sock_addrs():
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


def test_encode_fn_decode_fn():
    def g(x):
        return x

    def f():
        return g(42)

    assert decode_fn(encode_fn(f))() == f()


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
    stat = os.stat(zip_path)
    zip_inplace(str(tmpdir))
    assert os.stat(zip_path).st_mtime == stat.st_mtime
    zip_inplace(str(tmpdir), replace=True)
    assert os.stat(zip_path).st_mtime > stat.st_mtime


def test_static_default_dict():
    d = StaticDefaultDict({"foo": 42}, default=100500)
    assert d["foo"] == 42
    assert d["bar"] == 100500
    assert "bar" not in d


def test_env_create(tmpdir):
    env = PyEnv(
        name="test",
        python="{0.major}.{0.minor}".format(sys.version_info),
        pip_packages=["pycodestyle"])
    env_path = env.create(root=tmpdir)
    assert os.path.exists(env_path)
    assert os.path.isdir(env_path)

    env_python_bin = os.path.join(env_path, "bin", "python")
    check_output([env_python_bin, "-m", "pycodestyle", "--version"])
