from unittest import mock
from unittest.mock import ANY
import os

import pytest
from tensorflow_estimator.python.estimator.training import EvalSpec

from tf_yarn.tasks import evaluator_task
from tf_yarn.experiment import Experiment

import tensorflow as tf
from tensorflow.python import ops

from tf_yarn.tasks.evaluator_task import _get_step

checkpoints = {
   "/path/to/model/dir/model.ckpt-0",
   "/path/to/model/dir/model.ckpt-100",
   "/path/to/model/dir/model.ckpt-200",
   "/path/to/model/dir/model.ckpt-300"
}


@pytest.mark.parametrize("evaluated_ckpts,ckpt_to_export", [
    ({"/path/to/model/dir/model.ckpt-0", "/path/to/model/dir/model.ckpt-100",
      "/path/to/model/dir/model.ckpt-200"}, {"/path/to/model/dir/model.ckpt-300"}),
    (set(), {"/path/to/model/dir/model.ckpt-0", "/path/to/model/dir/model.ckpt-100",
     "/path/to/model/dir/model.ckpt-200", "/path/to/model/dir/model.ckpt-300"}),
    ({"/path/to/model/dir/model.ckpt-0", "/path/to/model/dir/model.ckpt-100",
      "/path/to/model/dir/model.ckpt-200", "/path/to/model/dir/model.ckpt-300"}, {})
])
def test_evaluate(evaluated_ckpts, ckpt_to_export):
    with mock.patch('tf_yarn._task_commons._get_experiment') as experiment_mock, \
            mock.patch('tf_yarn.tasks.evaluator_task._get_evaluated_checkpoint') \
            as _get_evaluated_checkpoint, \
            mock.patch('tf_yarn.tasks.evaluator_task._get_all_checkpoints') \
            as _get_checkpoints, \
            mock.patch('tf_yarn.tasks.evaluator_task.tf.io.gfile.exists') as exists_mock, \
            mock.patch('tf_yarn.tasks.evaluator_task.tf.io.gfile.listdir') as listdir_mock:
        exists_mock.side_effect = lambda *args, **kwargs: True
        listdir_mock.side_effect = lambda *args, **kwargs: evaluated_ckpts
        mock_exporter = mock.Mock(spec=tf.estimator.Exporter)
        mock_exporter.name = "my_best_exporter"

        mock_experiment = mock.Mock(spec=Experiment)
        mock_experiment.eval_spec = EvalSpec(
            mock.Mock(),
            exporters=mock_exporter,
            start_delay_secs=0,
            throttle_secs=0
        )
        mock_experiment.estimator.evaluate.side_effect = \
            lambda *args, **kwargs: {ops.GraphKeys.GLOBAL_STEP: 300}
        mock_experiment.estimator.model_dir = "model_dir"
        mock_experiment.train_spec.max_steps = 300

        experiment_mock.side_effect = lambda client: mock_experiment

        _get_evaluated_checkpoint.side_effect = lambda eval_dir: set(
            [_get_step(ckpt) for ckpt in evaluated_ckpts]
        )

        _get_checkpoints.side_effect = lambda model_dir: list(checkpoints)
        evaluator_task.evaluate(mock_experiment)

        assert len(mock_exporter.export.call_args_list) == len(ckpt_to_export)
        assert len(mock_experiment.estimator.evaluate.call_args_list) == len(ckpt_to_export)
        export_path = os.path.join(mock_experiment.estimator.model_dir, mock_exporter.name)
        if len(ckpt_to_export) > 0:
            for ckpt in ckpt_to_export:
                mock_exporter.export.assert_any_call(ANY, export_path, ckpt, ANY, ANY)
                mock_experiment.estimator.evaluate(
                    ANY, steps=ANY, hooks=ANY, name=ANY, checkpoint_path=ckpt
                )
