import contextlib
from unittest import mock
from unittest.mock import patch
import pytest

import cloudpickle
import skein

from tf_yarn._task_commons import _get_experiment


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
        stack.enter_context(patch(f'{MODULE_TO_TEST}.get_task_key'))
        mocked_event = stack.enter_context(patch(f'{MODULE_TO_TEST}.event'))
        mocked_client = mock.MagicMock(spec=skein.ApplicationClient)

        def experiment_f():
            raise Exception()

        mocked_client.kv.wait.return_value = cloudpickle.dumps(experiment_f)
        with pytest.raises(Exception):
            _get_experiment(mocked_client)
        mocked_event.start_event.assert_called_once()
        mocked_event.stop_event.assert_called_once()
