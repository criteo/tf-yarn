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
The cluster needs to know in advance how much resources to allocate for
each of the distributed TensorFlow task types.

The dataset in `examples/dnn_classification.py` is tiny and does not need
multi-node training. Therefore, it can be scheduled using just the `"chief"`
and `"evaluator"` tasks.

```python
from tf_skein import TaskSpec, YARNCluster

cluster = YARNCluster(task_specs={
    "chief": TaskSpec(memory=2 * 2**10, vcores=4),
    "evaluator": TaskSpec(memory=2**10, vcores=1)
})
```

The final step is to call the `run` method.

```python
cluster.run(experiment_fn, files={
    os.path.basename(winequality.__file__): winequality.__file__
})
```

Note that `run` allows to upload arbitrary files to the YARN containers.
Moreover, the uploaded Python modules and packages are automatically
importable. Refer to the API docs for more details.

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


### TensorFlow ⇆ YARN

`tf-skein` allocates a container for each distributed TensorFlow task. The
resources of the containers are configured separately for each task type.

### GPU/CPU

By default `tf-skein` allocates containers from the queue with CPU-only nodes.
To allocate a GPU-enabled container set the `queue` argument to
`YARNCluster.run` to `"ml-gpu"`:

```python
cluster = YARNCluster(...)
cluster.run(experiment_fn, queue="ml-gpu")
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
