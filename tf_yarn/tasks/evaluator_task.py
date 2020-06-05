import logging
import os
import time
from datetime import datetime, timedelta

import skein
import tensorflow as tf
from tensorflow.python import ops

from tf_yarn import _task_commons, event, metrics, cluster
from tf_yarn.tasks import logging as tf_yarn_logging

tf_yarn_logging.setup()

logger = logging.getLogger(__name__)


def evaluator_fn(client):
    experiment = _task_commons._get_experiment(client)
    evaluate(experiment, timeout_in_secs=1200)  # Timeout after 20min


def evaluate(experiment, stop_cond=None, timeout_in_secs=None):
    eval_dir = os.path.join(experiment.estimator.model_dir, "eval")
    if not tf.io.gfile.exists(eval_dir) or len(tf.io.gfile.listdir(eval_dir)) == 0:
        evaluated_checkpoints = set()
    else:
        evaluated_checkpoints = _get_evaluated_checkpoint(eval_dir)

    if len(evaluated_checkpoints) > 0:
        last_evaluated_checkpoint = max(evaluated_checkpoints)
        logger.info(f"Last evaluated checkpoint: {last_evaluated_checkpoint}")
        if experiment.train_spec.max_steps and \
                last_evaluated_checkpoint == experiment.train_spec.max_steps:
            logger.info(
                f"All checkpoints already evaluated; Max steps: {experiment.train_spec.max_steps}"
            )
            return

    time.sleep(experiment.eval_spec.start_delay_secs)
    timestamp = datetime.now()
    latest_checkpoint = False
    exporters = {}
    if experiment.eval_spec.exporters:
        exporters = {os.path.join(experiment.estimator.model_dir, exporter.name): exporter
                     for exporter in experiment.eval_spec.exporters}
    while not latest_checkpoint:
        if stop_cond and stop_cond():
            logger.info("Stop condition met")
            break
        if timeout_in_secs and datetime.now() > (timestamp + timedelta(seconds=timeout_in_secs)):
            logger.info("Stopping evaluation due to timeout")
            break
        all_checkpoint_steps = {
            ckpt for ckpt in _get_all_checkpoints(experiment.estimator.model_dir)
        }
        ckpt_to_eval = {
            ckpt for ckpt in all_checkpoint_steps
            if _get_step(ckpt) not in evaluated_checkpoints
        }

        for ckpt in ckpt_to_eval:
            timestamp = datetime.now()
            logger.info(f"Evaluating checkpoint {ckpt}")
            latest_eval_result = experiment.estimator.evaluate(
                experiment.eval_spec.input_fn,
                steps=experiment.eval_spec.steps,
                hooks=experiment.eval_spec.hooks,
                name=experiment.eval_spec.name,
                checkpoint_path=ckpt
            )
            evaluated_checkpoints.add(_get_step(ckpt))

            if experiment.train_spec.max_steps and latest_eval_result:
                global_step = latest_eval_result[ops.GraphKeys.GLOBAL_STEP]
                if global_step and global_step >= experiment.train_spec.max_steps:
                    logger.info("Last checkpoint evaluated")
                    latest_checkpoint = True

            for export_path, exporter in exporters.items():
                exporter.export(
                    experiment.estimator, export_path, ckpt, latest_eval_result, latest_checkpoint
                )

        if len(ckpt_to_eval) == 0:
            logger.info(f"No checkpoint to evaluate; Coming back to sleep "
                        f"({experiment.eval_spec.throttle_secs} secs)")

        time.sleep(experiment.eval_spec.throttle_secs)


def _get_step(checkpoint):
    return int(checkpoint.split("model.ckpt-")[1])


def _get_all_checkpoints(model_dir):
    return tf.train.get_checkpoint_state(model_dir).all_model_checkpoint_paths


def _get_evaluated_checkpoint(eval_dir):
    return set(metrics.get_all_metrics(eval_dir)['step'])


def main():
    client = skein.ApplicationClient.from_current()
    task = cluster.get_task()
    task_type, task_id = cluster.get_task_description()
    event.init_event(client, task, "127.0.0.1:0")
    _task_commons._setup_container_logs(client)

    if task_type == "evaluator":
        evaluator_fn(client)
    else:
        logger.info(f"{task_type}:{task_id}: nothing to do")

    event.stop_event(client, task, None)


if __name__ == '__main__':
    main()
