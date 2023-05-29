import logging
import os
import platform
import socket
from typing import (
    Optional,
    Tuple,
    List,
    Iterable,
    Iterator
)
from contextlib import contextmanager
from threading import Thread

from tf_yarn.topologies import ContainerTask

logger = logging.getLogger(__name__)

here = os.path.dirname(__file__)


class MonitoredThread(Thread):
    """A thread which captures any exception occurred during the
    execution of ``target``.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._exc = None

    @property
    def state(self):
        if self.is_alive():
            return "RUNNING"
        return "FAILED" if self.exception is not None else "SUCCEEDED"

    @property
    def exception(self) -> Optional[Exception]:
        return self._exc

    def run(self):
        try:
            super().run()
        except Exception as exc:
            self._exc = exc


def get_so_reuseport():
    try:
        return socket.SO_REUSEPORT
    except AttributeError:
        if platform.system() == "Linux":
            major, minor, *_ = platform.release().split(".")
            if (int(major), int(minor)) > (3, 9):
                # The interpreter must have been compiled on Linux <3.9.
                return 15
    return None


@contextmanager
def reserve_sock_addr() -> Iterator[Tuple[str, int]]:
    """Reserve an available TCP port to listen on.

    The reservation is done by binding a TCP socket to port 0 with
    ``SO_REUSEPORT`` flag set (requires Linux >=3.9). The socket is
    then kept open until the generator is closed.

    To reduce probability of 'hijacking' port, socket should stay open
    and should be closed _just before_ starting of ``tf.train.Server``
    """
    so_reuseport = get_so_reuseport()
    if so_reuseport is None:
        raise RuntimeError(
            "SO_REUSEPORT is not supported by the operating system") from None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, so_reuseport, 1)
        sock.bind(("", 0))
        _ipaddr, port = sock.getsockname()
        yield (socket.getfqdn(), port)


def iter_tasks(tasks: List[Tuple[str, int, int]]) -> Iterable[ContainerTask]:
    """Iterate the tasks in a TensorFlow cluster.
    """
    for task_type, n_instances, nb_process in tasks:
        yield from (ContainerTask(task_type, task_id, nb_process) for task_id in range(n_instances))


def xset_environ(**kwargs):
    """Exclusively set keys in the environment."""
    for key, value in kwargs.items():
        if key in os.environ:
            raise RuntimeError(f"{key} already set in os.environ: {value}")

    os.environ.update(kwargs)
