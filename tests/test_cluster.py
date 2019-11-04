import contextlib
import os
import typing
from unittest import mock

import pytest
import skein

from tf_yarn import cluster

MODULE_TO_TEST = "tf_yarn.cluster"


def test_aggregate_spec():
    client = mock.MagicMock(spec=skein.ApplicationClient)
    dict_sockaddr: typing.Dict[str, bytes] = {
        "worker:0/init": "1.1.1.1:8020".encode(),
        "worker:1/init": "1.1.1.2:4042".encode(),
        "ps:0/init": "1.1.1.3:8888".encode()
    }
    client.kv = mock.MagicMock(spec=skein.kv.KeyValueStore)
    client.kv.wait.side_effect = lambda arg: dict_sockaddr[arg]

    res = cluster.aggregate_spec(client, ["worker:1", "ps:0", "worker:0"])
    assert res == {"worker": ["1.1.1.1:8020", "1.1.1.2:4042"],
                   "ps": ["1.1.1.3:8888"]}


def test_get_task_description():
    with mock.patch.dict(os.environ):
        os.environ["SKEIN_CONTAINER_ID"] = "MYTASK_42"
        assert "MYTASK", 42 == cluster.get_task_description()


CURRENT_HOST = "1.1.1.1"
CURRENT_PORT = 8888
WORKER0_HOST = "1.1.1.2"
WORKER0_PORT = 8888


@pytest.mark.parametrize("task_name, task_index", [
    pytest.param("worker", 1),
    pytest.param("ps", 0)
])
def test_start_cluster_worker(task_name, task_index):
    task = f"{task_name}:{task_index}"

    CLUSTER_SPEC = {"worker:0/init": [f"{WORKER0_HOST}:{WORKER0_PORT}"],
                    f"{task}/init": [f"{CURRENT_HOST}:{CURRENT_PORT}"]}

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ))
        mock_event = stack.enter_context(mock.patch(f"{MODULE_TO_TEST}.event"))

        os.environ["SKEIN_CONTAINER_ID"] = f"{task_name}_{task_index}"

        mock_event.wait.side_effect = lambda client, key: CLUSTER_SPEC[key][0]
        mock_client = mock.Mock(spec=skein.ApplicationClient)
        cluster.start_cluster((CURRENT_HOST, CURRENT_PORT), mock_client, [task, "worker:0"])
        mock_event.init_event.assert_called_once_with(mock_client, task,
                                                      f"{CURRENT_HOST}:{CURRENT_PORT}")


@pytest.mark.parametrize("task_name, task_index, is_server_started", [
    pytest.param("worker", 1, True),
    pytest.param("ps", 0, False)
])
def test_start_tf_server(task_name, task_index, is_server_started):
    task = f"{task_name}:{task_index}"

    CLUSTER_SPEC = {"worker:0/init": [f"{WORKER0_HOST}:{WORKER0_PORT}"],
                    f"{task}/init": [f"{CURRENT_HOST}:{CURRENT_PORT}"]}

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ))
        os.environ["SKEIN_CONTAINER_ID"] = f"{task_name}_{task_index}"
        mock_server = stack.enter_context(mock.patch(f"{MODULE_TO_TEST}.tf.train"))
        cluster.start_tf_server(CLUSTER_SPEC)

        if is_server_started:
            assert mock_server.Server.call_count == 1
            _, kwargs = mock_server.Server.call_args
            assert kwargs["job_name"] == task_name
            assert kwargs["task_index"] == task_index
            assert kwargs["start"] is True
        else:
            assert mock_server.Server.call_count == 0
