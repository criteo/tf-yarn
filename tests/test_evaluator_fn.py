from unittest import mock
from unittest.mock import ANY
import os

import pytest
from tf_yarn.tasks import evaluator_fn
from tf_yarn.experiment import Experiment

import tensorflow as tf
from tensorflow.python import ops


checkpoints = {
   "/path/to/model/dir/model.ckpt-0",
   "/path/to/model/dir/model.ckpt-100",
   "/path/to/model/dir/model.ckpt-200",
   "/path/to/model/dir/model.ckpt-300"
}


@pytest.mark.parametrize("evaluated_ckpts,ckpt_to_export", [
    ({"/path/to/model/dir/model.ckpt-0", "/path/to/model/dir/model.ckpt-100",
      "/path/to/model/dir/model.ckpt-200"}, {"/path/to/model/dir/model.ckpt-300"}),
    ({}, {"/path/to/model/dir/model.ckpt-0", "/path/to/model/dir/model.ckpt-100",
     "/path/to/model/dir/model.ckpt-200", "/path/to/model/dir/model.ckpt-300"}),
    ({"/path/to/model/dir/model.ckpt-0", "/path/to/model/dir/model.ckpt-100",
      "/path/to/model/dir/model.ckpt-200", "/path/to/model/dir/model.ckpt-300"}, {})
])
def test_evaluate(evaluated_ckpts, ckpt_to_export):
    with mock.patch('tf_yarn._task_commons._get_experiment') as experiment_mock, \
            mock.patch('tf_yarn.tasks.evaluator_fn._get_evaluated_checkpoint') \
            as _get_evaluated_checkpoint, \
            mock.patch('tf_yarn.tasks.evaluator_fn._get_all_checkpoints') \
            as _get_checkpoints:
        mock_exporter = mock.Mock(spec=tf.estimator.Exporter)
        mock_exporter.name = "my_best_exporter"

        mock_experiment = mock.Mock(spec=Experiment)
        mock_experiment.eval_spec.start_delay_secs = 0
        mock_experiment.eval_spec.throttle_secs = 0
        mock_experiment.estimator.evaluate.side_effect = \
            lambda *args, **kwargs: {ops.GraphKeys.GLOBAL_STEP: 300}
        mock_experiment.estimator.model_dir = "model_dir"
        mock_experiment.train_spec.max_steps = 300
        mock_experiment.eval_spec.exporters = mock_exporter

        experiment_mock.side_effect = lambda client: mock_experiment

        _get_evaluated_checkpoint.side_effect = lambda eval_dir: list(evaluated_ckpts)

        _get_checkpoints.side_effect = lambda model_dir: list(checkpoints)
        evaluator_fn.evaluate(mock_experiment)

        assert len(mock_exporter.export.call_args_list) == len(ckpt_to_export)
        assert len(mock_experiment.estimator.evaluate.call_args_list) == len(ckpt_to_export)
        export_path = os.path.join(mock_experiment.estimator.model_dir, mock_exporter.name)
        if len(ckpt_to_export) > 0:
            for ckpt in ckpt_to_export:
                mock_exporter.export.assert_any_call(ANY, export_path, ckpt, ANY, ANY)
                mock_experiment.estimator.evaluate(
                    ANY, steps=ANY, hooks=ANY, name=ANY, checkpoint_path=ckpt
                )


@pytest.mark.parametrize("checkpoints,last_checkpoint", [
    (checkpoints, 300),
    ({"/path/to/model/dir/model.ckpt-300", "/path/to/model/dir/model.ckpt-0"}, 300),
    ({}, None)
])
def test_get_last_evaluated_checkpoint_steps(checkpoints, last_checkpoint):
    assert evaluator_fn._get_last_evaluated_checkpoint_steps(checkpoints) == last_checkpoint
