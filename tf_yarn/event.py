import logging
import time
import traceback
import typing

import skein

from tf_yarn.topologies import ContainerKey

_logger = logging.getLogger(__name__)


def wait(client: skein.ApplicationClient, key: str) -> str:
    """
    Wait for a key
    """
    _logger.info("Waiting for " + key)
    return client.kv.wait(key).decode()


def logs_event(client: skein.ApplicationClient,
               task_key: ContainerKey,
               logs: str) -> None:
    broadcast(client, f"{task_key.to_kv_str()}/logs", logs)


def url_event(client: skein.ApplicationClient,
              task_key: ContainerKey,
              url: str) -> None:
    broadcast(client, f"{task_key.to_kv_str()}/url", url)


def init_event(client: skein.ApplicationClient,
               task_key: ContainerKey,
               sock_addr: str) -> None:
    broadcast(client, f"{task_key.to_kv_str()}/init", sock_addr)


def start_event(client: skein.ApplicationClient,
                task_key: ContainerKey) -> None:
    broadcast(client, f"{task_key.to_kv_str()}/start")


def stop_event(client: skein.ApplicationClient,
               task_key: ContainerKey,
               e: typing.Optional[Exception]) -> None:
    broadcast(client, f"{task_key.to_kv_str()}/stop", maybe_format_exception(e))


def broadcast_train_eval_start_timer(client: skein.ApplicationClient,
                                     task_key: ContainerKey) -> None:
    broadcast(client, f'{task_key.to_kv_str()}/train_eval_start_time', str(time.time()))


def broadcast_train_eval_stop_timer(client: skein.ApplicationClient,
                                    task_key: ContainerKey) -> None:
    broadcast(client, f'{task_key.to_kv_str()}/train_eval_stop_time', str(time.time()))


def broadcast_container_start_time(client: skein.ApplicationClient,
                                   task_key: ContainerKey) -> None:
    broadcast(client, f'{task_key.to_kv_str()}/container_start_time', str(time.time()))


def broadcast_container_stop_time(client: skein.ApplicationClient,
                                  task_key: ContainerKey) -> None:
    broadcast(client, f'{task_key.to_kv_str()}/container_stop_time', str(time.time()))


def broadcast(
    client: skein.ApplicationClient,
    key: str,
    value: str = ""
) -> None:
    _logger.info(f"Broadcasting {key} = {value!r}")
    try:
        client.kv[key] = value.encode()
    except AttributeError:
        client.kv[key] = value


def maybe_format_exception(e: typing.Optional[Exception]) -> str:
    if e is None:
        return ""
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))
