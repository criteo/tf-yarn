import json
from unittest import mock
import traceback

import pytest
import skein
import tensorflow as tf

from tf_yarn.tensorflow.experiment import Experiment
from tf_yarn.tensorflow.keras_experiment import KerasExperiment
from tf_yarn.client import (
    _setup_cluster_spec,
    _setup_to_use_cuda_archive,
    get_safe_experiment_fn,
    SkeinCluster,
    run_on_yarn,
    ContainerLogStatus
)
from tf_yarn import constants
from tf_yarn.topologies import TaskSpec, ContainerKey

sock_addrs = {
    'chief': ['addr1:port1', 'addr10:port10', 'addr11:port11'],
    'evaluator': ['addr2:port2', 'addr3:port3'],
    'ps': ['addr4:port4', 'addr5:port5', 'addr6:port6'],
    'worker': ['addr7:port7', 'addr8:port8', 'addr9:port9']
}


@mock.patch("tf_yarn.client.skein.ApplicationClient")
@pytest.mark.parametrize("tasks_instances, expected_spec", [
    ([('chief', 1, 1), ('evaluator', 1, 1), ('ps', 1, 1), ('worker', 3, 1)],
     [['chief', 1, 1], ['ps', 1, 1], ['worker', 3, 1]]),
    ([('chief', 3, 1)], [['chief', 3, 1]]),
    ([('worker', 3, 1), ('ps', 3, 1)], [['worker', 3, 1], ['ps', 3, 1]]),
    ([('worker', 1, 1), ('evaluator', 0, 1)], [['worker', 1, 1]]),
    ([('worker', 1, 2), ('evaluator', 0, 1)], [['worker', 1, 2]])
])
def test_setup_cluster_spec(
        mock_skein_app,
        tasks_instances,
        expected_spec):
    kv_store = dict()
    mock_skein_app.kv = kv_store
    _setup_cluster_spec(
        tasks_instances,
        mock_skein_app
    )

    assert json.loads(kv_store[constants.KV_CLUSTER_INSTANCES].decode()) == expected_spec


def test_setup_to_use_cuda_archive():
    actual_pre_script_hook = _setup_to_use_cuda_archive(
        {"LD_LIBRARY_PATH": "cuda/usr/cuda-11.2/lib64"},
        "",
        "/user/prediction/cuda-runtimes/cuda-runtime-libs-11-2-2.tar.gz"
    )
    assert actual_pre_script_hook == \
        "hdfs dfs -get /user/prediction/cuda-runtimes/cuda-runtime-libs-11-2-2.tar.gz; \
         mkdir cuda; tar -xf cuda-runtime-libs-11-2-2.tar.gz -C ./cuda;"


def test_setup_to_use_cuda_archive_without_env():
    actual_pre_script_hook = _setup_to_use_cuda_archive(
        {},
        "",
        "/user/prediction/cuda-runtimes/cuda-runtime-libs-11-2-2.tar.gz"
    )
    assert actual_pre_script_hook == ""


def test_kill_skein_on_exception():
    def cloudpickle_raise_exception(*args, **kwargs):
        raise Exception("Cannot serialize your method!")

    with mock.patch('tf_yarn.client._setup_skein_cluster') as mock_setup_skein_cluster:
        with mock.patch('tf_yarn.client._setup_pyenvs'):
            with mock.patch('tf_yarn.client.cloudpickle.dumps') as mock_cloudpickle:
                mock_cloudpickle.side_effect = cloudpickle_raise_exception
                mock_app = mock.MagicMock(skein.ApplicationClient)
                mock_setup_skein_cluster.return_value = SkeinCluster(
                    client=None, app=mock_app,
                    event_listener=None, events=None,
                    tasks=[])
                try:
                    run_on_yarn(None, {}, pyenv_zip_path="/path/to/env")
                except Exception:
                    print(traceback.format_exc())
                    pass
                mock_app.shutdown.assert_called_once_with(
                    skein.model.FinalStatus.FAILED)


def _experiment_fn(model_dir):
    print(f"create experiment with model_dir={model_dir}")

    def model_fn():
        return tf.estimator.EstimatorSpec()

    def train_fn():
        return None

    def eval_fn():
        return None

    return Experiment(
        tf.estimator.LinearClassifier(
            feature_columns=[], model_dir=model_dir,
            loss_reduction=tf.keras.losses.Reduction.SUM
        ),
        tf.estimator.TrainSpec(train_fn),
        tf.estimator.EvalSpec(eval_fn))


def _keras_experiment_fn(model_dir):
    print(f"create Keras experiment with model_dir={model_dir}")

    model = tf.keras.Sequential()

    return KerasExperiment(
        model=model,
        model_dir=model_dir,
        train_params=None,
        input_data_fn=None,
        target_data_fn=None,
        validation_data_fn=None)


def test_get_safe_experiment_fn():
    with mock.patch('importlib.import_module') as mock_import_module:
        module = mock.Mock()
        module.experiment_fn = _experiment_fn
        mock_import_module.return_value = module
        experiment_fn = get_safe_experiment_fn("testpackage.testmodule.experiment_fn",
                                               "test_model_dir")
        print(f"got function .. {experiment_fn}")
        print("execute function ..")
        print(experiment_fn)
        experiment = experiment_fn()
        print(experiment)
        assert isinstance(experiment, Experiment) is True
        assert experiment.estimator.model_dir == "test_model_dir"
        mock_import_module.assert_called_once_with("testpackage.testmodule")


def test_get_safe_keras_experiment_fn():
    _keras_experiment_fn("test_model_dir")
    with mock.patch('importlib.import_module') as mock_import_module:
        module = mock.Mock()
        module.experiment_fn = _keras_experiment_fn
        mock_import_module.return_value = module
        experiment_fn = get_safe_experiment_fn("testpackage.testmodule.experiment_fn",
                                               "test_model_dir")
        print(f"got function .. {experiment_fn}")
        print("execute function ..")
        print(experiment_fn)
        experiment = experiment_fn()
        print(experiment)
        assert isinstance(experiment, KerasExperiment) is True
        assert experiment.model_dir == "test_model_dir"
        mock_import_module.assert_called_with("testpackage.testmodule")


@pytest.mark.parametrize("nb_retries,nb_failures", [(0, 0), (1, 0), (1, 1), (2, 2)])
def test_retry_run_on_yarn(nb_retries, nb_failures):
    cpt = 0

    def fail(*args, **kwargs):
        if cpt < nb_failures:
            raise Exception("")
        else:
            pass

    with mock.patch('tf_yarn.client._setup_pyenvs'), \
            mock.patch('tf_yarn.client._setup_skein_cluster') as mock_setup_skein_cluster, \
            mock.patch('tf_yarn.client._run_on_cluster') as mock_run_on_cluster:
        mock_run_on_cluster.side_effect = fail

        gb = 2**10

        try:
            run_on_yarn(
                lambda: Experiment(None, None, None),
                task_specs={
                    "chief": TaskSpec(memory=16 * gb, vcores=16),
                    "worker": TaskSpec(memory=16 * gb, vcores=16, instances=1),
                    "ps": TaskSpec(memory=16 * gb, vcores=16, instances=1)
                },
                pyenv_zip_path="path/to/env",
                nb_retries=nb_retries
            )
        except Exception:
            pass

        nb_calls = min(nb_retries, nb_failures) + 1
        assert mock_run_on_cluster.call_count == nb_calls
        assert mock_setup_skein_cluster.call_count == nb_calls


def test_container_log_status():
    container_log_status = ContainerLogStatus(
        {ContainerKey("chief", 0):
            ("http://ec-0d-9a-00-3a-c0.pa4.hpc.criteo.preprod:8042/node/"
                "containerlogs/container_e17294_1569204305368_264801_01_000002/myuser"),
         ContainerKey("evaluator", 0):
            ("http://ec-0d-9a-00-3a-c0.pa4.hpc.criteo.preprod:8042/node/"
                "containerlogs/container_e95614_6456565654646_344343_01_000003/myuser")},
        {ContainerKey("chief", 0): "SUCCEEDED", ContainerKey("evaluator", 0): "FAILED"}
    )

    containers = container_log_status.by_container_id()

    assert containers["container_e17294_1569204305368_264801_01_000002"] == \
           (ContainerKey("chief", 0), "SUCCEEDED")
    assert containers["container_e95614_6456565654646_344343_01_000003"] == \
           (ContainerKey("evaluator", 0), "FAILED")
