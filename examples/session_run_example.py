"""
Example of using low level API

It should :
* Create a yarn application with 2 containers called 'worker'
* The graph is distributed from this client to these workers
* Compute the mean
"""
import sys

import numpy as np
import tensorflow as tf

from tf_yarn import event
from tf_yarn import TaskSpec, TFYarnExecutor

"""
You need to package tf-yarn in order to ship it to the executors
First create a pex from root dir
pex . -o examples/tf-yarn.pex
"""
PEX_FILE = "tf-yarn.pex"

NODE_NAME = "worker"


def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    print(sys.argv)

    with TFYarnExecutor(pyenv_zip_path=f"{PEX_FILE}",
                        standalone_client_mode=True) as tfYarnExecutor:
        cluster = tfYarnExecutor.setup_skein_cluster(
            task_specs={
                NODE_NAME: TaskSpec(memory=4 * 2**10, vcores=32, instances=2)
            }
        )

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

        first_task = cluster.cluster_spec.job_tasks(NODE_NAME)[1]
        with tf.Session(f"grpc://{first_task}",
                        config=tf.ConfigProto(operation_timeout_in_ms=300000)) as sess:
            result = sess.run(mean, feed_dict={x: np.random.random(size)})
            print(f"mean = {result}")

        event.broadcast(cluster.app, "stop", "1")


if __name__ == "__main__":
    main()
