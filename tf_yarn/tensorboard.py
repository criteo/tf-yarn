import os
import threading
import socket

import skein
from tensorboard import default
from tensorboard import program

from tf_yarn import event, cluster, Experiment
from . import _internal


DEFAULT_TERMINATION_TIMEOUT_SECONDS = 30


def get_termination_timeout():
    timeout = os.environ.get('SERVICE_TERMINATION_TIMEOUT_SECONDS')
    if timeout is not None:
        timeout = int(timeout)
    else:
        timeout = DEFAULT_TERMINATION_TIMEOUT_SECONDS  # Set the default timeout
    return timeout


def start_tf_board(client: skein.ApplicationClient,
                   experiment: Experiment = None):
    thread = None
    if experiment:
        model_dir = experiment.estimator.config.model_dir
    else:
        model_dir = os.environ.get('TF_BOARD_MODEL_DIR', None)
    task = cluster.get_task()
    os.environ['GCS_READ_CACHE_DISABLED'] = '1'
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'cpp'
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION'] = '2'
    try:
        program.setup_environment()
        tensorboard = program.TensorBoard(default.get_plugins(),
                                          default.get_assets_zip_provider())
        with _internal.reserve_sock_addr() as (h, p):
            tensorboard_url = f"http://{h}:{p}"
            argv = ['tensorboard', f"--logdir={model_dir}",
                    f"--port={p}"]
            # Append more arguments if needed.
            if 'TF_BOARD_EXTRA_ARGS' in os.environ:
                argv += os.environ['TF_BOARD_EXTRA_ARGS'].split(' ')
            tensorboard.configure(argv)
        tensorboard.launch()
        event.url_event(client, task, f"Tensorboard is listening at {tensorboard_url}")
        thread = [t for t in threading.enumerate() if t.name == 'TensorBoard'][0]
    except Exception as e:
        event.stop_event(client, task, e)

    return thread
