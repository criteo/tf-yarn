import os
import shutil
import socket
import typing
from base64 import b64encode, b64decode
from contextlib import contextmanager
from threading import Thread

import dill


class MonitoredThread(Thread):
    """A thread which captures any exception occurred during the
    execution of ``target``.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._exc = None

    def exception(self) -> typing.Optional[Exception]:
        return self._exc

    def run(self):
        try:
            super().run()
        except Exception as exc:
            self._exc = exc


@contextmanager
def reserve_sock_addr() -> typing.ContextManager[typing.Tuple[str, int]]:
    """Reserve an available TCP port to listen on.

    The acquired TCP socket is hold open until the generator is
    closed. This does not eliminate the chance of collision between
    multiple concurrent Python processes, but it makes it slightly
    less likely.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        _ipaddr, port = sock.getsockname()
        yield (socket.gethostname(), port)


def encode_fn(fn) -> str:
    """Encode a function in a plain-text format."""
    return b64encode(dill.dumps(fn, recurse=True)).decode()


def decode_fn(s: str):
    """Decode a function encoded by ``encode_fn``."""
    return dill.loads(b64decode(s))


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
            os.rename(created, zip_path)
        except OSError as e:
            os.remove(created)  # Cleanup on failure.
            raise e from None
    return zip_path


def spec_from_kv(
    kv,
    stage: str,
    num_workers: int,
    num_ps: int
) -> typing.Dict[str, list]:
    def get(target):
        return kv.wait(stage + "/" + target).decode()

    spec = {
        "chief": [get("chief_0")]
    }

    for idx in range(num_ps):
        spec.setdefault("ps", []).append(get(f"ps_{idx}"))

    for idx in range(num_workers):
        spec.setdefault("worker", []).append(get(f"worker_{idx}"))

    return spec


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
