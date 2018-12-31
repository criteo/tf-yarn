import time
import traceback

import skein
import typing
import tensorflow as tf


def wait(client: skein.ApplicationClient, key: str) -> str:
    """
    Wait for a key
    """
    tf.logging.info("Waiting for " + key)
    return client.kv.wait(key).decode()


def logs_event(client: skein.ApplicationClient,
               task: str,
               logs: str) -> None:
    broadcast(client, f"{task}/logs", logs)


def url_event(client: skein.ApplicationClient,
              task: str,
              url: str) -> None:
    broadcast(client, f"{task}/url", url)


def init_event(client: skein.ApplicationClient,
               task: str,
               sock_addr: str) -> None:
    broadcast(client, f"{task}/init", sock_addr)


def start_event(client: skein.ApplicationClient,
                task: str) -> None:
    broadcast(client, f"{task}/start")


def stop_event(client: skein.ApplicationClient,
               task: str,
               e: typing.Optional[Exception]) -> None:
    broadcast(client, f"{task}/stop", maybe_format_exception(e))


def broadcast_train_eval_start_timer(client: skein.ApplicationClient,
                                     task: str) -> None:
    broadcast(client, f'{task}/train_eval_start_time', str(time.time()))


def broadcast_train_eval_stop_timer(client: skein.ApplicationClient,
                                    task: str) -> None:
    broadcast(client, f'{task}/train_eval_stop_time', str(time.time()))


def broadcast_container_start_time(client: skein.ApplicationClient,
                                   task: str) -> None:
    broadcast(client, f'{task}/container_start_time', str(time.time()))


def broadcast_container_stop_time(client: skein.ApplicationClient,
                                  task: str) -> None:
    broadcast(client, f'{task}/container_stop_time', str(time.time()))


def broadcast(
    client: skein.ApplicationClient,
    key: str,
    value: str = ""
) -> None:
    tf.logging.info(f"Broadcasting {key} = {value!r}")
    try:
        client.kv[key] = value.encode()
    except AttributeError:
        client.kv[key] = value


def maybe_format_exception(e: typing.Optional[Exception]) -> str:
    if e is None:
        return ""
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))
