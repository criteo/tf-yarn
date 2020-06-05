# tf-yarnᵝ

tf-yarn is a Python library we have built at Criteo for training TensorFlow models on a Hadoop/YARN cluster. An introducing blog post can be found [here](https://medium.com/criteo-labs/train-tensorflow-models-on-yarn-in-just-a-few-lines-of-code-ba0f354f38e3).

It supports running on one worker or on multiple workers with different distribution strategies and it can run on CPUs or GPUs using just a few lines of code.

Its API provides an easy entry point for working with Estimators. Keras is currently supported via the [model_to_estimator](https://www.tensorflow.org/api_docs/python/tf/keras/estimator/model_to_estimator) conversion function. Please refer to the [examples](https://github.com/criteo/tf-yarn/tree/master/examples) for some code samples.

[MLflow](https://www.mlflow.org/docs/latest/quickstart.html) is supported for all kind of trainings (one worker/distributed).
More infos [here](https://github.com/criteo/tf-yarn/blob/master/docs/MLflow.md).

[Tensorboard](https://github.com/criteo/tf-yarn/blob/master/docs/Tensorboard.md) can be spawned in a separate container during learnings.

Two alternatives to TensorFlow's distribution strategies are available:
[Horovod with gloo](https://github.com/criteo/tf-yarn/blob/master/docs/HorovodWithGloo.md) and [tf-collective-all-reduce](https://github.com/criteo/tf-collective-all-reduce)

![tf-yarn](https://github.com/criteo/tf-yarn/blob/master/skein.png?raw=true)

## Installation

### Install with Pip

```bash
$ pip install tf-yarn
```

### Install from source

```bash
$ git clone https://github.com/criteo/tf-yarn
$ cd tf-yarn
$ pip install .
```

### Prerequisites

tf-yarn only supports Python ≥3.6.

Make sure to have Tensorflow working with HDFS by setting up all the environment variables as described [here](https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/hadoop.md).

You can run the `check_hadoop_env` script to check that your setup is OK (it has been installed by tf_yarn):

```
$ check_hadoop_env
# You should see something like
# INFO:tf_yarn.bin.check_hadoop_env:results will be written in /home/.../shared/Dev/tf-yarn/check_hadoop_env.log
# INFO:tf_yarn.bin.check_hadoop_env:check_env: True
# INFO:tf_yarn.bin.check_hadoop_env:write dummy file to hdfs hdfs://root/tmp/a1df7b99-fa47-4a86-b5f3-9bc09019190f/hello_tf_yarn.txt
# INFO:tf_yarn.bin.check_hadoop_env:check_local_hadoop_tensorflow: True
# INFO:root:Launching remote check
# ...
# INFO:tf_yarn.bin.check_hadoop_env:remote_check: True
# INFO:tf_yarn.bin.check_hadoop_env:Hadoop setup: OK
```

### run_on_yarn

The only abstraction tf-yarn adds on top of the ones already present in
TensorFlow is `experiment_fn`. It is a function returning a triple of one `Estimator` and two specs -- `TrainSpec` and `EvalSpec`.

Here is a stripped down `experiment_fn` from one of the provided [examples][linear_classifier_example] to give you an idea of how it might look:

```python
from tf_yarn import Experiment

def experiment_fn():
  # ...
  estimator = tf.estimator.DNNClassifier(...)
  return Experiment(
    estimator,
    tf.estimator.TrainSpec(train_input_fn, max_steps=...),
    tf.estimator.EvalSpec(eval_input_fn)
 )
```

An experiment can be scheduled on YARN using the run_on_yarn function which takes three required arguments:

- `pyenv_zip_path` which contains the tf-yarn modules and dependencies like TensorFlow to be shipped to the cluster. pyenv_zip_path can be generated easily with a helper function based on the current installed virtual environment;
- `experiment_fn` as described above;
- `task_specs` dictionary specifying how much resources to allocate for each of the distributed TensorFlow task type.

The example uses the [Wine Quality][wine-quality] dataset from UCI ML repository. With just under 5000 training instances available, there is no need for multi-node training, meaning that a chief complemented by an evaluator would manage just fine. Note that each task will be executed in its own YARN container.

```python
from tf_yarn import TaskSpec, run_on_yarn
import cluster_pack

pyenv_zip_path, _ = cluster_pack.upload_env()
run_on_yarn(
    pyenv_zip_path,
    experiment_fn,
    task_specs={
        "chief": TaskSpec(memory="2 GiB", vcores=4),
        "evaluator": TaskSpec(memory="2 GiB", vcores=1),
        "tensorboard": TaskSpec(memory="2 GiB", vcores=1)
    }
)
```

The final bit is to forward the `winequality.py` module to the YARN containers,
in order for the tasks to be able to import them:

```python
run_on_yarn(
    ...,
    files={
        os.path.basename(winequality.__file__): winequality.__file__,
    }
)
```

Under the hood, the experiment function is shipped to each container, evaluated and then passed to the `train_and_evaluate` function.

```python
experiment = experiment_fn()
tf.estimator.train_and_evaluate(
  experiment.estimator,
  experiment.train_spec,
  experiment.eval_spec
)
```

[linear_classifier_example]: https://github.com/criteo/tf-yarn/blob/master/examples/linear_classifier_example.py
[wine-quality]: https://archive.ics.uci.edu/ml/datasets/Wine+Quality

## Distributed TensorFlow

The following is a brief summary of the core distributed TensorFlow concepts relevant to training [estimators](https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate) with the ParameterServerStrategy, as it is the distribution strategy activated by default when training Estimators on multiple nodes.

Distributed TensorFlow operates in terms of tasks.
A task has a type which defines its purpose in the distributed TensorFlow cluster:
- `worker` tasks headed by the `chief` doing model training
- `chief` task additionally handling checkpoints, saving/restoring the model, etc.
- `ps`  tasks (aka parameter servers) storing the model itself. These tasks typically do not compute anything.
Their sole purpose is serving the model variables
- `evaluator` task periodically evaluating the model from the saved checkpoint

The types of tasks can depend on the distribution strategy, for example, ps tasks are only used by ParameterServerStrategy.
The following picture presents an example of a cluster setup with 2 workers, 1 chief, 1 ps and 1 evaluator.

```
+-----------+              +---------+   +----------+   +----------+
| evaluator |        +-----+ chief:0 |   | worker:0 |   | worker:1 |
+-----+-----+        |     +----^----+   +-----^----+   +-----^----+
      ^              |          |            |              |
      |              v          |            |              |
      |        +-----+---+      |            |              |
      |        | model   |   +--v---+        |              |
      +--------+ exports |   | ps:0 <--------+--------------+
               +---------+   +------+
```

The cluster is defined by a ClusterSpec, a mapping from task types to their associated network addresses. For instance, for the above example, it looks like that:

```
{
  "chief": ["chief.example.com:2125"],
  "worker": ["worker0.example.com:6784",
             "worker1.example.com:6475"],
  "ps": ["ps0.example.com:7419"],
  "evaluator": ["evaluator.example.com:8347"]
}
```
Starting a task in the cluster requires a ClusterSpec. This means that the spec should be fully known before starting any of the tasks.

Once the cluster is known, we need to export the ClusterSpec through the [TF_CONFIG](https://cloud.google.com/ml-engine/docs/tensorflow/distributed-training-details) environment variable and start the TensorFlow server on each container.

Then we can run the [train-and-evaluate](https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate) function on each container.
We just launch the same function as in local training mode, TensorFlow will automatically detect that we have set up a ClusterSpec and start a distributed learning.

You can find more information about distributed Tensorflow [here](https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/distributed.md) and about distributed training Estimators [here](https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate).

## Training with multiple workers

Activating the previous example in tf-yarn is just changing the cluster_spec by adding the additional `worker` and `ps` instances: 

```python
run_on_yarn(
    ...,
    task_specs={
        "chief": TaskSpec(memory="2 GiB", vcores=4),
        "worker": TaskSpec(memory="2 GiB", vcores=4, instances=2),
        "ps": TaskSpec(memory="2 GiB", vcores=8),
        "evaluator": TaskSpec(memory="2 GiB", vcores=1),
        "tensorboard": TaskSpec(memory="2 GiB", vcores=1)
    }
)
```

## Configuring the Python interpreter and packages

tf-yarn uses [cluster-pack](https://github.com/criteo/cluster-pack) to to ship an isolated virtual environment to the containers.
(You should have installed the dependencies from `requirements.txt` into your virtual environment first `pip install -r requirements.txt`)
This works if you use Anaconda and also with [Virtual Environments](https://docs.python.org/3/tutorial/venv.html).

By default the generated package is a [pex][pex] package. cluster-pack will generate the pex package, upload it to hdfs and you can start tf_yarn by providing the hdfs path.

```python
import cluster_pack
pyenv_zip_path, env_name = cluster_pack.upload_env()
run_on_yarn(
    pyenv_zip_path=pyenv_zip_path
)
```

If you hosting evironment is Anaconda `upload_env` the packaging module will use [conda-pack][conda-pack] to create the package.

You can also directly use the command line tools provided by [conda-pack][conda-pack] and [pex][pex] to generate the packages.

For pex you can run this command in the root directory to create the package (it includes all requirements from setup.py)
```
pex . -o myarchive.pex
```

You can then run tf-yarn with your generated package:

```python
run_on_yarn(
    pyenv_zip_path="myarchive.pex"
)
```

[conda-pack]: https://conda.github.io/conda-pack/
[pex]: https://pex.readthedocs.io/en/stable/

## Running on GPU

YARN does not have first-class support for GPU resources. A common workaround is
to use [node labels][node-labels] where CPU-only nodes are unlabelled, while
the GPU ones have a label. Furthermore, in this setting GPU nodes are
typically bound to a separate queue which is different from the default one.

Currently, tf-yarn assumes that the GPU label is ``"gpu"``. There are no
assumptions on the name of the queue with GPU nodes, however, for the sake of
example we wil use the name ``"ml-gpu"``.

The default behaviour of `run_on_yarn` is to run on CPU-only nodes. In order
to run on the GPU ones:

1. Set the `queue` argument.
2. Set `TaskSpec.label` to `NodeLabel.GPU` for relevant task types.
   A good rule of a thumb is to run compute heavy `"chief"` and `"worker"`
   tasks on GPU, while keeping `"ps"` and `"evaluator"` on CPU.

```python
import getpass
import cluster_pack
from tf_yarn import NodeLabel


pyenv_zip_path, _ = cluster_pack.upload_env()
run_on_yarn(
    pyenv_zip_path
    experiment_fn,
    task_specs={
        "chief": TaskSpec(memory="2 GiB", vcores=4, label=NodeLabel.GPU),
        "evaluator": TaskSpec(memory="1 GiB", vcores=1)
    },
    queue="ml-gpu"
)
```
The previous example applies to TensorFlow >= 1.15.
For TensorFlow < 1.15 you need to call upload_env with tensorflow-gpu package and provide it to `run_on_yarn`.

[node-labels]: https://hadoop.apache.org/docs/stable/hadoop-yarn/hadoop-yarn-site/NodeLabel.html

## Accessing HDFS in the presence of [federation][federation]

`skein` the library underlying `tf_yarn` automatically acquires a delegation token
for ``fs.defaultFS`` on security-enabled clusters. This should be enough for most
use-cases. However, if your experiment needs to access data on namenodes other than
the default one, you have to explicitly list them in the `file_systems` argument
to `run_on_yarn`. This would instruct `skein` to acquire a delegation token for
these namenodes in addition to ``fs.defaultFS``:

```python
run_on_yarn(
    ...,
    file_systems=["hdfs://preprod"]
)
```

Depending on the cluster configuration, you might need to point libhdfs to a
different configuration folder. For instance:

```python
run_on_yarn(
    ...,
    env={"HADOOP_CONF_DIR": "/etc/hadoop/conf.all"}
)
```

[federation]: https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/Federation.html

## Running model evaluation independently

Model training and model evaluation can be run independently. To do so, you must
use parameter `custom_task_module` of `run_on_yarn`.

To run model training without evaluation:
```python
run_on_yarn(
    ...,
    task_specs={
        "chief": TaskSpec(memory="2 GiB", vcores=4),
        "worker": TaskSpec(memory="2 GiB", vcores=4, instances=2),
        "ps": TaskSpec(memory="2 GiB", vcores=8),
        "tensorboard": TaskSpec(memory="2 GiB", vcores=1)
    }
)
```

To run model evaluation:
```python
run_on_yarn(
    ...,
    task_specs={
        "evaluator": TaskSpec(memory="2 GiB", vcores=1)
    },
    custom_task_module="tf_yarn.tasks.evaluator_task"
)
```
