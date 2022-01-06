import time
import os
import logging.config
from typing import Union, List

import tensorflow as tf
import skein

from tf_yarn.event import broadcast
from tf_yarn.tensorflow import experiment, keras_experiment
from tf_yarn import mlflow
from tf_yarn._task_commons import n_try, is_chief, get_task


logger = logging.getLogger(__name__)


class StepPerSecondHook(tf.estimator.StepCounterHook):

    def __init__(
        self,
        every_n_steps=100,
        every_n_secs=None,
        output_dir=None,
        summary_writer=None
    ):
        tf.estimator.StepCounterHook.__init__(
            self,
            every_n_steps=every_n_steps,
            every_n_secs=every_n_secs,
            output_dir=output_dir,
            summary_writer=summary_writer
        )

    def _log_and_record(self, elapsed_steps: int, elapsed_time: float, global_step: int):
        if is_chief():
            steps_per_sec = elapsed_steps / elapsed_time
            mlflow.log_metric(f"steps_per_sec_{n_try()}", steps_per_sec, step=global_step)


class EvalMonitorHook(tf.estimator.SessionRunHook):
    '''
    Hook to generate statistics about evaluator usage
    Usage: tf.estimator.EvalSpec(...,hooks=[Eval_analyzer_hook()])
    '''
    def __init__(self):
        self.client = skein.ApplicationClient.from_current()
        self.task = get_task()
        self.step_counter = 0
        self.eval_start_time = 0.0
        self.eval_step_dur_accu = 0.0
        self.start_time = time.time()

    def before_run(self, run_context):
        self.eval_start_time = time.time()
        return tf.estimator.SessionRunArgs(tf.compat.v1.train.get_global_step())

    def after_run(self, _run_context, run_values):
        self.step_counter += 1
        cur_time = time.time()
        self.eval_step_dur_accu += cur_time - self.eval_start_time
        self.broadcast('eval_step_mean_duration', str(self.eval_step_dur_accu / self.step_counter))
        self.broadcast(
            'awake_time_ratio',
            str(self.eval_step_dur_accu / (cur_time - self.start_time))
        )
        self.broadcast('nb_eval_steps', str(self.step_counter))
        self.broadcast('last_training_step', str(run_values.results))

    def broadcast(self, key: str, value: str):
        broadcast(self.client, f'{self.task}/{key}', value)


def get_all_metrics(model_path):
    events = _gen_events_iterator(model_path)
    dataframe = {
        'step': list(),
        'name': list(),
        'value': list()
    }
    for event in events:
        summary = event.summary
        if summary:
            for value in summary.value:
                if value.simple_value:
                    dataframe['step'].append(event.step)
                    dataframe['name'].append(value.tag)
                    dataframe['value'].append(value.simple_value)
    return dataframe


def _is_event_file(filename):
    return os.path.basename(filename).startswith('events.out')


def _gen_events_iterator(model_path):
    event_file = next((filename for filename in tf.compat.v1.gfile.ListDirectory(model_path)
                       if _is_event_file(filename)))
    assert isinstance(event_file, str)
    return tf.compat.v1.train.summary_iterator(os.path.join(model_path, event_file))


def _hook_name_already_exists(
        hook: tf.estimator.SessionRunHook,
        hooks: List[tf.estimator.SessionRunHook]) -> bool:
    hook_name = type(hook).__name__
    return len([h for h in hooks
                if type(h).__name__ == hook_name]) > 0


def _add_monitor_to_experiment(
    my_experiment: Union[experiment.Experiment, keras_experiment.KerasExperiment]
) -> Union[experiment.Experiment, keras_experiment.KerasExperiment]:
    if isinstance(my_experiment, experiment.Experiment):
        logger.info(f"configured training hooks: {my_experiment.train_spec.hooks}")

        training_hooks = list(my_experiment.train_spec.hooks)

        if my_experiment.config.log_step_count_steps is not None:
            steps_per_second_hook = StepPerSecondHook(
                every_n_steps=my_experiment.config.log_step_count_steps
            )
            if not _hook_name_already_exists(steps_per_second_hook, training_hooks):
                training_hooks.append(steps_per_second_hook)
            else:
                logger.warning("do not add StepPerSecondHook as there is already one configured")

        monitored_train_spec = my_experiment.train_spec._replace(
            hooks=training_hooks
        )

        monitored_eval_spec = my_experiment.eval_spec._replace(
            hooks=(EvalMonitorHook(), *my_experiment.eval_spec.hooks)
        )

        my_experiment = my_experiment._replace(
            eval_spec=monitored_eval_spec, train_spec=monitored_train_spec)
    elif isinstance(my_experiment, keras_experiment.KerasExperiment):
        logger.warning("equivalent of StepPerSecondHook not yet implemented for KerasExperiment")
    else:
        raise ValueError("experiment must be an Experiment or a KerasExperiment")
    return my_experiment
