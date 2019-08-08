"""
Example of using low level API

It should :
* Create a yarn application with 2 containers called 'worker'
* The graph is distributed from this client to these workers
* Compute the mean
"""
import sys

import logging
import numpy as np
import skein
import tensorflow as tf

from tf_yarn import event
from tf_yarn import TaskSpec, packaging, standalone_client_mode

NODE_NAME = "worker"

logging.basicConfig(level="INFO")

logger = logging.getLogger(__name__)


def main():
    pyenv_zip_path, env_name = packaging.upload_env_to_hdfs()
    editable_requirements = packaging.get_editable_requirements_from_current_venv()
    session_config = tf.ConfigProto(operation_timeout_in_ms=300000)
    with standalone_client_mode(
            pyenv_zip_path,
            task_specs={
                NODE_NAME: TaskSpec(memory="2 GiB", vcores=4, instances=2)
            },
            tf_session_config=session_config,
            files={
                **editable_requirements,
            }
    ) as cluster_spec:
        size = 10000
        x = tf.placeholder(tf.float32, size)

        with tf.device(f"/job:{NODE_NAME}/task:1"):
            with tf.name_scope("scope_of_task1"):
                first_batch = tf.slice(x, [5000], [-1])
                mean1 = tf.reduce_mean(first_batch)

        with tf.device(f"/job:{NODE_NAME}/task:0"):
            with tf.name_scope("scope_of_task0"):
                second_batch = tf.slice(x, [0], [5000])
                mean2 = tf.reduce_mean(second_batch)
                mean = (mean1 + mean2) / 2

        cluster_spec_dict = cluster_spec.as_dict()
        first_task = next(iter(cluster_spec_dict.values()))[0]
        logger.info("cluster_spec:" + str(cluster_spec_dict))
        logger.info("connecting to target:" + first_task)

        with tf.Session(f"grpc://{first_task}", config=session_config) as sess:
            result = sess.run(mean, feed_dict={x: np.random.random(size)})
            print(f"mean = {result}")


if __name__ == "__main__":
    main()
