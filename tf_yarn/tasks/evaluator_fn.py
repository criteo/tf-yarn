import logging
import os
from datetime import datetime, timedelta
from tensorflow.python import ops
import time
from threading import Thread
import tensorflow as tf
from tf_yarn.tasks import logging as tf_yarn_logging
from tf_yarn import _task_commons, event, metrics
tf_yarn_logging.setup()

logger = logging.getLogger(__name__)


def evaluator_fn(client):
    def _daemon(stop):
        experiment = _task_commons._get_experiment(client)
        evaluate(experiment, stop)

    stop_evaluation = False
    thread = Thread(target=_daemon, args=(lambda: stop_evaluation,), daemon=True)
    thread.start()

    event.wait(client, "chief:0/stop")
    stop_evaluation = True


def evaluate(experiment, stop_cond=None, timeout_in_secs=None):
    eval_dir = os.path.join(experiment.estimator.model_dir, "eval")
    evaluated_checkpoints = _get_evaluated_checkpoint(eval_dir)

    if len(evaluated_checkpoints) > 0:
        last_evaluated_checkpoint = \
            _get_last_evaluated_checkpoint_steps(evaluated_checkpoints)
        if experiment.train_spec.max_steps and \
                last_evaluated_checkpoint == experiment.train_spec.max_steps:
            return

    time.sleep(experiment.eval_spec.start_delay_secs)
    timestamp = datetime.now()
    latest_checkpoint = False
    exporters = {}
    if experiment.eval_spec.exporters:
        exporters = experiment.eval_spec.exporters \
            if isinstance(experiment.eval_spec.exporters, list) \
            else [experiment.eval_spec.exporters]
        exporters = {os.path.join(experiment.estimator.model_dir, exporter.name): exporter
                     for exporter in exporters}
    while not latest_checkpoint:
        if stop_cond and stop_cond():
            break
        if timeout_in_secs and datetime.now() > (timestamp + timedelta(seconds=timeout_in_secs)):
            break
        all_checkpoint_steps = {
            ckpt for ckpt in _get_all_checkpoints(experiment.estimator.model_dir)
        }
        ckpt_to_eval = {ckpt for ckpt in all_checkpoint_steps if ckpt not in evaluated_checkpoints}
        for ckpt in ckpt_to_eval:
            timestamp = datetime.now()
            latest_eval_result = experiment.estimator.evaluate(
                experiment.eval_spec.input_fn,
                steps=experiment.eval_spec.steps,
                hooks=experiment.eval_spec.hooks,
                name=experiment.eval_spec.name,
                checkpoint_path=ckpt
            )
            evaluated_checkpoints.add(ckpt)

            if experiment.train_spec.max_steps and latest_eval_result:
                global_step = latest_eval_result[ops.GraphKeys.GLOBAL_STEP]
                if global_step and global_step >= experiment.train_spec.max_steps:
                    latest_checkpoint = True

            for export_path, exporter in exporters.items():
                exporter.export(
                    experiment.estimator, export_path, ckpt, latest_eval_result, latest_checkpoint
                )

        time.sleep(experiment.eval_spec.throttle_secs)


def _get_all_checkpoints(model_dir):
    return tf.train.get_checkpoint_state(model_dir).all_model_checkpoint_paths()


def _get_evaluated_checkpoint(eval_dir):
    return set(metrics.get_all_metrics(eval_dir)['step'])


def _get_last_evaluated_checkpoint_steps(evaluated_checkpoints):
    if len(evaluated_checkpoints) == 0:
        return None
    return sorted([int(ckpt.split("model.ckpt-")[1])
                   for ckpt in evaluated_checkpoints])[-1]
