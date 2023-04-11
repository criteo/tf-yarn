import contextlib
from unittest import mock
from unittest.mock import patch
import pytest

import cloudpickle
import skein

from tf_yarn._task_commons import _get_experiment, choose_master, MASTER_ADDR, MASTER_PORT


MODULE_TO_TEST = "tf_yarn._task_commons"


def test__get_experiment_object():
    mocked_client = mock.MagicMock(spec=skein.ApplicationClient)
    experiment_obj = 'obj'

    def experiment_f():
        return experiment_obj

    mocked_client.kv.wait.return_value = cloudpickle.dumps(experiment_f)
    returned_object = _get_experiment(mocked_client)
    assert returned_object == experiment_obj


def test__get_experiment_exception():
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch(f'{MODULE_TO_TEST}.get_task'))
        mocked_event = stack.enter_context(patch(f'{MODULE_TO_TEST}.event'))
        mocked_client = mock.MagicMock(spec=skein.ApplicationClient)

        def experiment_f():
            raise Exception()

        mocked_client.kv.wait.return_value = cloudpickle.dumps(experiment_f)
        with pytest.raises(Exception):
            _get_experiment(mocked_client)
        mocked_event.start_event.assert_called_once()
        mocked_event.stop_event.assert_called_once()


def test_choose_master_rank_0():
    with mock.patch(f"{MODULE_TO_TEST}._internal") as internal_mock:
        addr = "127.0.0.1"
        port = 1313
        internal_mock.reserve_sock_addr.return_value.__enter__.return_value = (addr, port)
        kv_store = dict()
        skein_client = mock.Mock()
        skein_client.kv = kv_store
        master_addr, master_port = choose_master(skein_client, 0)
        assert kv_store.get(MASTER_ADDR, None) == addr.encode()
        assert kv_store.get(MASTER_PORT, None) == str(port).encode()
        assert master_addr == addr
        assert master_port == port


def test_choose_master():
    addr = "127.0.0.1"
    port = 1313
    kv_store = {
        MASTER_ADDR: addr.encode(),
        MASTER_PORT: str(port).encode()
    }
    skein_client = mock.Mock()
    skein_client.kv.wait = kv_store.get
    master_addr, master_port = choose_master(skein_client, 1)
    assert master_addr == addr
    assert master_port == port
