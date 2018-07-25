import errno
import logging
import os
import shutil
import socket
import typing
from base64 import b64encode, b64decode
from contextlib import ExitStack
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


def iter_available_sock_addrs():
    """Iterate available TCP ports to listen on.

    The acquired TCP sockets are hold open until the generator is
    closed. This does not eliminate the chance of collision between
    multiple concurrent Python processes, but it makes it slightly
    less likely.
    """
    with ExitStack() as stack:
        host = socket.gethostname()
        while True:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            stack.enter_context(s)
            try:
                s.bind(("", 0))
            except socket.error as e:
                if e.errno == errno.EADDRINUSE:
                    continue
                else:
                    raise

            _ipaddr, port = s.getsockname()
            yield f"{host}:{port}"


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
    assert os.path.exists(path)
    assert os.path.isdir(path)

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


class KVBarrier:
    def __init__(self, kv, stage: str, num_workers: int, num_ps: int) -> None:
        self.kv = kv
        self.stage = stage
        self.num_workers = num_workers
        self.num_ps = num_ps
        self.logger = logging.getLogger(self.__class__.__name__)

    def wait(self, key: str, value: str = ""):
        self.logger.info(f"Entering {self.stage} barrier")
        self.kv[self.stage + "/" + key] = value
        self.logger.info(f"Written {key} = {value}")

        def get(target):
            if target == key:
                return

            self.logger.info("Waiting for " + target)
            return self.kv.wait(self.stage + "/" + target)

        spec = {
            "chief": [get("chief:0")]
        }

        for idx in range(self.num_ps):
            spec.setdefault("ps", []).append(get(f"ps:{idx}"))

        for idx in range(self.num_workers):
            spec.setdefault("worker", []).append(get(f"worker:{idx}"))

        return spec
