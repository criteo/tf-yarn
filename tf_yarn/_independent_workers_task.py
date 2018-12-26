# Copyright 2018 Criteo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from tf_yarn._task_commons import (
    _prepare_container, _execute_dispatched_function,
    _shutdown_container, _process_arguments, _get_experiment
)
from . import cluster


def main() -> None:
    task_type, task_id = cluster.get_task_description()
    client, cluster_spec, cluster_tasks = _prepare_container()

    # Variable TF_CONFIG must be set before instantiating
    # the estimator to train in a distributed way
    cluster.setup_tf_config(cluster_spec)
    experiment = _get_experiment(client)
    run_config = experiment.config

    tf.logging.info(f"Starting server {task_type}:{task_id}")
    cluster.start_tf_server(cluster_spec, run_config.session_config)
    thread = _execute_dispatched_function(client, experiment)

    # "ps" tasks do not terminate by themselves. See
    # https://github.com/tensorflow/tensorflow/issues/4713.
    if task_type != "ps":
        thread.join()
        tf.logging.info(f"{task_type}:{task_id} {thread.state}")

    _shutdown_container(client, cluster_tasks, run_config, thread)


if __name__ == "__main__":
    _process_arguments()
    main()
