import pytest

from tf_yarn._dispatch_task import matches_device_filters


@pytest.mark.parametrize("task,device_filters", [
    ("ps:0", ["/job:ps", "/job:worker/task:42"]),
    ("worker:42", ["/job:ps", "/job:worker/task:42"])
])
def test_matches_device_filters(task, device_filters):
    assert matches_device_filters(task, device_filters)


@pytest.mark.parametrize("task,device_filters", [
    ("chief:0", ["/job:ps", "/job:worker/task:42"]),
    ("worker:0", ["/job:ps", "/job:worker/task:42"]),
    ("evaluator:0", ["/job:ps", "/job:worker/task:42"])
])
def test_does_not_match_device_filters(task, device_filters):
    assert not matches_device_filters(task, device_filters)
