import logging
import os
from psutil import net_if_addrs
from socket import AddressFamily
import warnings

import skein
import tensorflow as tf

try:
    import horovod.tensorflow as hvd
    from horovod.run import gloo_run
    from horovod.run.http.http_server import RendezvousServer
except (ModuleNotFoundError, ImportError) as e:
    warnings.warn("Horovod is not installed. See README for instructions to install it")
    raise e

from tf_yarn import event, _task_commons
from tf_yarn.tensorflow import Experiment, KerasExperiment
from tf_yarn.tensorflow.tasks.evaluator_task import evaluator_fn
from tf_yarn._task_commons import (
    setup_logging, get_task_key, is_chief
)
setup_logging()

logger = logging.getLogger(__name__)

N_PROCESS_PER_WORKER = 1


def get_net_if():
    return [(key, value[0].address) for key, value in net_if_addrs().items()
            if value[0].family == AddressFamily.AF_INET and key != 'lo'][0]


def _worker_fn(client, task, net_if):
    event.broadcast(client, f"{task}/addr", net_if[1])

    worker_info = event.wait(client, f"chief:0/{net_if[1]}").split(',')
    driver_socket = event.wait(client, "chief:0/sock_addr").split(':')

    os.environ['HOROVOD_GLOO_RENDEZVOUS_ADDR'] = driver_socket[0]
    os.environ['HOROVOD_GLOO_RENDEZVOUS_PORT'] = driver_socket[1]
    os.environ['HOROVOD_CONTROLLER'] = 'gloo'
    os.environ['HOROVOD_CPU_OPERATIONS'] = 'gloo'
    os.environ['HOROVOD_GLOO_IFACE'] = net_if[0]
    os.environ['HOROVOD_RANK'] = worker_info[0]
    os.environ['HOROVOD_SIZE'] = worker_info[1]
    os.environ['HOROVOD_LOCAL_RANK'] = worker_info[2]
    os.environ['HOROVOD_LOCAL_SIZE'] = worker_info[3]
    os.environ['HOROVOD_CROSS_RANK'] = worker_info[4]
    os.environ['HOROVOD_CROSS_SIZE'] = worker_info[5]

    hvd.init()

    experiment = _task_commons._get_experiment(client)

    if isinstance(experiment, Experiment):
        if not is_chief(task.type):
            # Overwrite config to do nothing but training to improve training speed
            experiment.estimator._model_dir = "."
            new_config = experiment.estimator.config.replace(
                save_summary_steps=None,
                save_checkpoints_steps=None,
                save_checkpoints_secs=None,
                log_step_count_steps=None
            )
            experiment.estimator._config = new_config

        logger.info("start training..")

        experiment.estimator.train(
            experiment.train_spec.input_fn,
            hooks=experiment.train_spec.hooks,
            max_steps=experiment.train_spec.max_steps)
    elif isinstance(experiment, KerasExperiment):
        if not is_chief(task.type):
            if experiment.train_params['callbacks'] is not None:
                callbacks_to_keep = []
                for callback in experiment.train_params['callbacks']:
                    if not isinstance(callback, tf.keras.callbacks.ModelCheckpoint):
                        callbacks_to_keep.append(callback)
                experiment.train_params['callbacks'] = callbacks_to_keep
        if experiment.input_data_fn is not None:
            experiment.train_params['x'] = experiment.input_data_fn()
        if experiment.target_data_fn is not None:
            experiment.train_params['y'] = experiment.target_data_fn()
        logger.info("start training..")
        experiment.model.fit(**experiment.train_params)
    else:
        raise ValueError("experiment must be an Experiment or a KerasExperiment")


def _driver_fn(client, net_if):
    cluster_tasks = _task_commons._get_cluster_tasks(client)
    # Worker discovery

    # TODO : fetch n_procvess_per_worker from cluster_tasks
    worker_list = [f"{net_if[1]}:{N_PROCESS_PER_WORKER}"]
    n_workers = 1
    for cluster_task in cluster_tasks:
        if 'worker' in cluster_task:
            worker_addr = event.wait(client, f"{cluster_task.to_container_key().to_kv_str()}/addr")
            logger.info(f"{cluster_task}: {worker_addr}")
            worker_list.append(f"{worker_addr}:{N_PROCESS_PER_WORKER}")
            n_workers += 1

    # Worker task allocation to workers
    # ToDO: check if gloo_run needs to strip the nb_process_per_worker from cluster_tasks
    hosts = gloo_run.parse_hosts(','.join(worker_list))
    host_alloc_plan = gloo_run.get_host_assignments(hosts, n_workers)
    for host in host_alloc_plan:
        host_info = f"""\
            {host.rank},{host.size},{host.local_rank},\
            {host.local_size},{host.cross_rank},{host.cross_size}\
            """
        event.broadcast(client, f"{get_task_key()}/{host.hostname}", host_info)

    global_rendezv = RendezvousServer(verbose=1)
    global_rendezv_port = global_rendezv.start_server()
    global_rendezv.httpd.init(host_alloc_plan)
    event.broadcast(client, f"{get_task_key()}/sock_addr", f"{net_if[1]}:{global_rendezv_port}")
    return global_rendezv.listen_thread


def main():
    client = skein.ApplicationClient.from_current()
    task = get_task_key()
    event.init_event(client, task, "127.0.0.1:0")
    _task_commons._setup_container_logs(client)
    net_if = get_net_if()

    if task.type == 'chief':
        _driver_fn(client, net_if)
    if task.type in ['worker', 'chief']:
        _worker_fn(client, task, net_if)
    elif task.type == 'evaluator':
        evaluator_fn(client)
    else:
        logger.error(f'Unknown task type {task.type}')

    event.stop_event(client, task, None)


if __name__ == '__main__':
    main()
