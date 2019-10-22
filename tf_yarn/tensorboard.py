import logging
import os
import threading
import socket

import skein
import tensorflow as tf
from tensorboard import program

from typing import Optional, Iterable, Tuple

from tf_yarn import _internal, event, cluster

_logger = logging.getLogger(__name__)


DEFAULT_TERMINATION_TIMEOUT_SECONDS = 30
URL_EVENT_LABEL = "Tensorboard listening on"


def get_termination_timeout():
    timeout = os.environ.get('TB_TERMINATION_TIMEOUT_SECONDS')
    if timeout is not None:
        timeout = int(timeout)
    else:
        timeout = DEFAULT_TERMINATION_TIMEOUT_SECONDS  # Set the default timeout
    return timeout


def start_tf_board(client: skein.ApplicationClient, tf_board_model_dir: str):
    task = cluster.get_task()
    os.environ['GCS_READ_CACHE_DISABLED'] = '1'
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'cpp'
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION'] = '2'
    try:
        program.setup_environment()
        tensorboard = program.TensorBoard()
        with _internal.reserve_sock_addr() as (h, p):
            tensorboard_url = f"http://{h}:{p}"
            argv = ['tensorboard', f"--logdir={tf_board_model_dir}",
                    f"--port={p}"]
            tb_extra_args = os.getenv('TB_EXTRA_ARGS', "")
            if tb_extra_args:
                argv += tb_extra_args.split(' ')
            tensorboard.configure(argv)
        tensorboard.launch()
        event.start_event(client, task)
        event.url_event(client, task, f"{tensorboard_url}")
    except Exception as e:
        _logger.error("Cannot start tensorboard", e)
        event.stop_event(client, task, e)


def url_event_name(tasks: Iterable[str]) -> Optional[str]:
    tensorboard_tasks = [t for t in tasks
                         if t.startswith('tensorboard')]
    if len(tensorboard_tasks) == 1:
        return tensorboard_tasks[0] + "/url"

    return None
