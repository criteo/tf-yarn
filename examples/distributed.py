"""
Example of using low level API

Create first a pex with tf yarn call example.pex
pex tf-yarn -o example.pex

Then execute the script 'python distributed.py'

It should :
* Create a yarn application with 2 containers called 'worker'
* The graph is distributed from this client to these workers
* Compute the mean
"""

import sys

import skein
import numpy as np
import tensorflow as tf

from tf_yarn import cluster
from tf_yarn import event

NODE_NAME = "worker"


def create_cluster():
    client = skein.ApplicationClient.from_current()
    cluster_spec = cluster.start_cluster(client, [f'{NODE_NAME}:0', f'{NODE_NAME}:1'])
    cluster.start_tf_server(cluster_spec)
    event.wait(client, "stop")


def create_skein_app():
    service = skein.Service(['./example.pex distributed.py --server'],
                            skein.Resources(2*1024, 1),
                            env={'PEX_ROOT': '/tmp/{uuid.uuid4()}/'},
                            files={'example.pex': 'example.pex',
                                   'distributed.py': __file__},
                            instances=2)
    spec = skein.ApplicationSpec(
            {NODE_NAME: service},
            queue='dev')
    return spec


def client_tf(client):
    spec = create_skein_app()
    app = client.submit_and_connect(spec)
    x = tf.placeholder(tf.float32, 100)

    with tf.device(f"/job:{NODE_NAME}/task:1"):
        first_batch = tf.slice(x, [0], [50])
        mean1 = tf.reduce_mean(first_batch)

    with tf.device(f"/job:{NODE_NAME}/task:0"):
        second_batch = tf.slice(x, [50], [-1])
        mean2 = tf.reduce_mean(second_batch)
        mean = (mean1 + mean2) / 2

    first_task = event.wait(app, f"{NODE_NAME}:0/init")
    with tf.Session(f"grpc://{first_task}") as sess:
        result = sess.run(mean, feed_dict={x: np.random.random(100)})
        print(f"mean = {result}")
    event.broadcast(app, "stop", "1")


def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    print(sys.argv)
    if '--server' in sys.argv:
        create_cluster()
    else:
        with skein.Client() as client:
            client_tf(client)


if __name__ == "__main__":
    main()
