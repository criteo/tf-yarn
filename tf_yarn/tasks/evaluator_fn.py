import logging
from tensorflow.python import ops
import time
from threading import Thread
from tf_yarn.tasks import logging as tf_yarn_logging
from tf_yarn import _task_commons, event
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


def evaluate(experiment, stop_cond=None):
    time.sleep(experiment.eval_spec.start_delay_secs)
    evaluated_checkpoints = set()
    while True:
        latest_checkpoint = experiment.estimator.latest_checkpoint()
        latest_eval_result = None
        if latest_checkpoint and latest_checkpoint not in evaluated_checkpoints:
            latest_eval_result = experiment.estimator.evaluate(
                experiment.eval_spec.input_fn,
                steps=experiment.eval_spec.steps,
                hooks=experiment.eval_spec.hooks,
                name=experiment.eval_spec.name
            )
            evaluated_checkpoints.add(latest_checkpoint)

            if experiment.train_spec.max_steps:
                if latest_eval_result:
                    global_step = latest_eval_result[ops.GraphKeys.GLOBAL_STEP]
                    if global_step and global_step >= experiment.train_spec.max_steps:
                        break
            else:
                if stop_cond and stop_cond():
                    break

        time.sleep(experiment.eval_spec.throttle_secs)
