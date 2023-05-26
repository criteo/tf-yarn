
import contextlib
import os
from unittest import mock
from unittest.mock import patch
import json

import pytest
import skein
import tensorflow as tf

from tf_yarn._internal import iter_tasks, MonitoredThread
from tf_yarn.tensorflow.tasks.tf_task_common import (
    _prepare_container, _matches_device_filters, _execute_dispatched_function,
    _wait_for_connected_tasks, _shutdown_container
)
from tf_yarn.tensorflow import Experiment
from tf_yarn.topologies import ContainerTask, ContainerKey

MODULE_TO_TEST = "tf_yarn.tensorflow.tasks.tf_task_common"


@pytest.mark.parametrize("task,device_filters", [
    (ContainerTask("ps", 0, 1), ["/job:ps", "/job:worker/task:42"]),
    (ContainerTask("worker", 42, 2), ["/job:ps", "/job:worker/task:42"])
])
def test_matches_device_filters(task, device_filters):
    assert _matches_device_filters(task, device_filters)


@pytest.mark.parametrize("task,device_filters", [
    (ContainerTask("chief", 0, 1), ["/job:ps", "/job:worker/task:42"]),
    (ContainerTask("worker", 0, 2), ["/job:ps", "/job:worker/task:42"]),
    (ContainerTask("evaluator", 0, 1), ["/job:ps", "/job:worker/task:42"])
])
def test_does_not_match_device_filters(task, device_filters):
    assert not _matches_device_filters(task, device_filters)


def test__prepare_container():
    with contextlib.ExitStack() as stack:
        # mock modules
        mocked_client_call = stack.enter_context(
            patch(f"{MODULE_TO_TEST}.skein.ApplicationClient.from_current"))
        mocked_logs = stack.enter_context(patch(f'{MODULE_TO_TEST}._setup_container_logs'))
        mocked_cluster_spec = stack.enter_context(patch(f'{MODULE_TO_TEST}.cluster.start_cluster'))

        # fill client mock
        mocked_client = mock.MagicMock(spec=skein.ApplicationClient)
        host_port = ('localhost', 1234)
        instances = [('worker', 10, 2), ('chief', 1, 1)]
        mocked_client.kv.wait.return_value = json.dumps(instances).encode()
        mocked_client_call.return_value = mocked_client
        (client, cluster_spec, cluster_tasks) = _prepare_container(host_port)

        # checks
        mocked_logs.assert_called_once()
        mocked_cluster_spec.assert_called_once_with(host_port, mocked_client, cluster_tasks)
        assert client == mocked_client
        assert cluster_tasks == list(iter_tasks(instances))


def test__execute_dispatched_function():
    with contextlib.ExitStack() as stack:
        mocked_event = stack.enter_context(patch(f'{MODULE_TO_TEST}.event'))
        mocked_train = stack.enter_context(
            patch(f'{MODULE_TO_TEST}.tf.estimator.train_and_evaluate'))
        passed_args = []
        mocked_train.side_effect = lambda *args: passed_args.append(args)
        mocked_task_key = \
            stack.enter_context(patch(f'{MODULE_TO_TEST}.get_task_key'))
        mocked_task_key.return_value = ContainerKey("worker", 0)

        mocked_client = mock.MagicMock(spec=skein.ApplicationClient)
        mocked_experiment = Experiment(None, None, None)
        thread = _execute_dispatched_function(mocked_client, mocked_experiment)
        # assert thread.state == 'RUNNING'
        thread.join()
        mocked_event.start_event.assert_called_once()
        assert passed_args == [(None, None, None)]
        assert thread.state == 'SUCCEEDED'


def test_wait_for_connected_tasks():
    with contextlib.ExitStack() as stack:
        mocked_event = stack.enter_context(patch(f'{MODULE_TO_TEST}.event'))
        mocked_filter = stack.enter_context(patch(f'{MODULE_TO_TEST}._matches_device_filters'))
        mocked_filter.return_value = True
        tasks = [ContainerTask('task', 1, 1), ContainerTask('task', 2, 1)]
        message = 'tag'
        _wait_for_connected_tasks(None, tasks, None, message)
        calls = [mock.call(
            None, f'{task.to_container_key().to_kv_str()}/{message}') for task in tasks]
        mocked_event.wait.assert_has_calls(calls, any_order=True)


def test__shutdown_container():
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch(f'{MODULE_TO_TEST}.cluster'))
        mocked_event = stack.enter_context(patch(f'{MODULE_TO_TEST}.event'))
        mocked_wait = stack.enter_context(patch(f'{MODULE_TO_TEST}._wait_for_connected_tasks'))

        mocked_session_config = mock.MagicMock(spec=tf.compat.v1.ConfigProto)
        mocked_thread = mock.MagicMock(spec=MonitoredThread)
        mocked_thread.exception.return_value = Exception()
        os.environ["SKEIN_CONTAINER_ID"] = "worker_0"
        with pytest.raises(Exception):
            _shutdown_container(None, None, mocked_session_config, mocked_thread)

        mocked_event.stop_event.assert_called_once()
        mocked_wait.assert_called_once()
