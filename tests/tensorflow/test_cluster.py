import contextlib
import os
import typing
from unittest import mock

import pytest
import skein

from tf_yarn.tensorflow import cluster
from tf_yarn._task_commons import get_task_description

MODULE_TO_TEST = "tf_yarn.tensorflow.cluster"


def test_aggregate_spec():
    client = mock.MagicMock(spec=skein.ApplicationClient)
    dict_sockaddr: typing.Dict[str, bytes] = {
        "worker:0:2/init": "1.1.1.1:8020".encode(),
        "worker:1:2/init": "1.1.1.2:4042".encode(),
        "ps:0:1/init": "1.1.1.3:8888".encode()
    }
    client.kv = mock.MagicMock(spec=skein.kv.KeyValueStore)
    client.kv.wait.side_effect = lambda arg: dict_sockaddr[arg]

    res = cluster.aggregate_spec(client, ["worker:1:2", "ps:0:1", "worker:0:2"])
    assert res == {"worker": ["1.1.1.1:8020", "1.1.1.2:4042"],
                   "ps": ["1.1.1.3:8888"]}


def test_get_task_description():
    with mock.patch.dict(os.environ):
        os.environ["SKEIN_CONTAINER_ID"] = "MYTASK_42"
        assert "MYTASK", 42 == get_task_description()


CURRENT_HOST = "1.1.1.1"
CURRENT_PORT = 8888
WORKER0_HOST = "1.1.1.2"
WORKER0_PORT = 8888
WORKER1_HOST = "1.1.1.3"
WORKER1_PORT = 8888


@pytest.mark.parametrize("task_name, task_index, n_process_per_instance", [
    pytest.param("worker", 1, 2),
    pytest.param("ps", 0, 1)
])
def test_start_cluster_worker(task_name, task_index, n_process_per_instance):
    task = f"{task_name}:{task_index}:{n_process_per_instance}"

    CLUSTER_SPEC = {"worker:0:1/init": [f"{WORKER0_HOST}:{WORKER0_PORT}"],
                    f"{task}/init": [f"{CURRENT_HOST}:{CURRENT_PORT}"]}

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ))
        mock_event = stack.enter_context(mock.patch(f"{MODULE_TO_TEST}.event"))

        os.environ["SKEIN_CONTAINER_ID"] = f"{task_name}_{task_index}"

        mock_event.wait.side_effect = lambda client, key: CLUSTER_SPEC[key][0]
        mock_client = mock.Mock(spec=skein.ApplicationClient)
        cluster.start_cluster((CURRENT_HOST, CURRENT_PORT), mock_client, [task, "worker:0:1"])
        mock_event.init_event.assert_called_once_with(mock_client, f"{task_name}:{task_index}",
                                                      f"{CURRENT_HOST}:{CURRENT_PORT}")


@pytest.mark.parametrize("task_name, task_index, is_server_started", [
    pytest.param("worker", 1, True),
    pytest.param("ps", 0, False)
])
def test_start_tf_server(task_name, task_index, is_server_started):

    CLUSTER_SPEC = {"worker": [f"worker0.{WORKER0_HOST}:{WORKER0_PORT}:1",
                               f"worker1.{WORKER1_HOST}:{WORKER1_PORT}:2"],
                    "ps": [f"ps0.{CURRENT_HOST}:{CURRENT_PORT}:1"]}

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ))
        os.environ["SKEIN_CONTAINER_ID"] = f"{task_name}_{task_index}"
        mock_server = stack.enter_context(mock.patch(f"{MODULE_TO_TEST}.tf.distribute"))
        cluster.start_tf_server(CLUSTER_SPEC)

        if is_server_started:
            assert mock_server.Server.call_count == 1
            _, kwargs = mock_server.Server.call_args
            assert kwargs["job_name"] == task_name
            assert kwargs["task_index"] == task_index
            assert kwargs["start"] is True
        else:
            assert mock_server.Server.call_count == 0
