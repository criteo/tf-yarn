import errno
import socket
import typing
from threading import Thread
from contextlib import ExitStack


class MonitoredThread(Thread):
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


def _iter_available_sock_addrs():
    """Iterate available TCP ports to listen on.

    The iterator holds a reference to all the acquired TCP ports until
    it is closed. This does not eliminate the chance of collision
    between multiple concurrent Python processes, but it makes it
    slightly less likely.
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


def _spec_from_iter(
    reserved: typing.Iterator[str],
    num_workers: int,
    num_ps: int
):
    spec = {
        "chief": [next(reserved)],
    }

    for _ in range(num_workers):
        spec.setdefault("worker", []).append(next(reserved))
    for _ in range(num_ps):
        spec.setdefault("ps", []).append(next(reserved))
    return spec


def _spec_from_kv(kv, num_workers: int, num_ps: int):
    spec = {
        "chief": [kv.wait("chief:0")]
    }

    for idx in range(num_ps):
        spec.setdefault("ps", []).append(kv.wait(f"ps:{idx}"))

    for idx in range(num_workers):
        spec.setdefault("worker", []).append(kv.wait(f"worker:{idx}"))

    return spec
