tf-skein
========

<img src="https://gitlab.criteois.com/s.lebedev/tf-skein/raw/master/skein.png"
    width="40%" />


Installation
------------

```bash
$ pip install git+https://gitlab.criteois.com/s.lebedev/tf-skein.git
```


Quickstart
----------

The core abstraction in `tf-skein` is called an `ExperimentFn`. It is
a function returning a triple of an `Estimator`, and two specs --
`TrainSpec` and `EvalSpec`.

Here is a stripped down `experiment_fn` from
[`examples/dnn_classification.py`](examples/dnn_classification.py)
to give you an idea of how it might look:

``` python
from tf_skein import Experiment

def experiment_fn():
    # ...
    estimator = tf.estimator.DNNClassifier(...)
    return Experiment(
        estimator,
        tf.estimator.TrainSpec(train_input_fn),
        tf.estimator.EvalSpec(eval_input_fn)
```

Having an `experiment_fn` we can run it on YARN using a `YARNCluster`.
The cluster needs to know in advance the environment it will be working it.
The environment consists of

* Python interpreter and packages, see `Env.MINIMAL_CPU` and `Env.MINIMAL_GPU`;
* local files to be uploaded, and
* environment variables to be forwarded.

```python
from tf_skein import YARNCluster

cluster = YARNCluster(files={
    os.path.basename(winequality.__file__): winequality.__file__
})
```

The final step is to call the `run` method with an `experiment_fn` and
a dictionary specifying how much resources to allocate for each of the
distributed TensorFlow task types. The dataset in
`examples/dnn_classification.py` is tiny and does not need multi-node
training. Therefore, it can be scheduled using just the `"chief"` and
`"evaluator"` tasks. Each task will be executed in its own container.

```python
from tf_skein import TaskSpec

cluster.run(experiment_fn, task_specs={
    "chief": TaskSpec(memory=2 * 2**10, vcores=4),
    "evaluator": TaskSpec(memory=2**10, vcores=1)
})
```

### Distributed TensorFlow 101

This is a brief summary of the core distributed TensorFlow concepts. Please
refer to the [official documentation][distributed-tf] for the full version.

Distributed TensorFlow operates in terms of tasks. A task has a type which
defines its purpose in the distributed TensorFlow cluster. ``"worker"`` tasks
headed by the `"chief"` do model training. The `"chief"` additionally handles
checkpointing, saving/restoring the model, etc. The model itself is stored
on one or more `"ps"` tasks. These tasks typically do not compute anything.
Their sole purpose is serving the variables of the model. Finally, the
`"evaluator"` task is responsible for periodically evaluating the model.

At the minimum, the cluster must have a single `"chief"` task. However, it
is a good idea to complement it by the `"evaluator"` to allow for running
the evaluation in parallel with the training.

```
+-----------+              +-------+   +----------+   +----------+
| evaluator |        +-----+ chief |   | worker:0 |   | worker:1 |
+-----+-----+        |     +----^--+   +-----^----+   +-----^----+
      ^              |          |            |              |
      |              v          |            |              |
      |        +-----+---+      |            |              |
      |        | model   |   +--v---+        |              |
      +--------+ exports |   | ps:0 <--------+--------------+
               +---------+   +------+
```

### Setting up the Python environment

`Env` specifies the Python environment shipped to the containers. `tf-skein`
comes with two predefined environments: `Env.MINIMAL_CPU` (default) and
`Env.MINIMAL_GPU` which differ in the TensorFlow build the use: CPU- or
GPU-enabled.

Additional pip-installable packages can be added via the `Env.extended_with`
method. Note that the method returns a *new* environment.

```python
from tf_skein import Env

keras_gpu_env = Env.MINIMAL_GPU.extended_with("keras_gpu_env", packages=[
    "keras==2.2.0"
])
```

### Running on GPU@Criteo

By default `YARNCluster` runs an experiment on CPU-only nodes. To run on GPU
on the pa4.preprod cluster:

1. Set the `"queue"` argument to `YARNCluster.run` to `"ml-gpu"`.
2. Set `TaskSpec.node_label` to `"gpu"` for relevant task types. In general,
   it is a good idea to run compute heavy `"chief"`, `"worker"` tasks on GPU,
   while keeping `"ps"` and `"evaluator"` on CPU.

Example:

```python
cluster.run(experiment_fn, queue="ml-gpu", task_specs={
    "chief": TaskSpec(memory=2 * 2**10, vcores=4, node_label="gpu"),
    "evaluator": TaskSpec(memory=2**10, vcores=1)
})
```

Limitations
-----------

`tf-skein` uses [Miniconda][miniconda] for creating relocatable
Python environments. The package management, however, is done by
pip to allow for more flexibility. The downside to that is that
it is impossible to create an environment for an OS/architecture
different from the one the library is running on.

[miniconda]: https://conda.io/miniconda.html
[tf-estimators]: https://www.tensorflow.org/guide/estimators
[distributed-tf]: https://www.tensorflow.org/deploy/distributed
[skein]: https://jcrist.github.io/skein
[skein-tutorial]: https://jcrist.github.io/skein/quickstart.html
