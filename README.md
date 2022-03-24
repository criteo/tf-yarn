# tf-yarn

![tf-yarn](https://github.com/criteo/tf-yarn/blob/master/skein.png?raw=true)

tf-yarn is a Python library we have built at Criteo for training Pytorch and TensorFlow models on a Hadoop/YARN cluster. An introducing blog post can be found [here](https://medium.com/criteo-labs/train-tensorflow-models-on-yarn-in-just-a-few-lines-of-code-ba0f354f38e3).

It supports mono and multi-worker training, different distribution strategies and can run on CPUs or GPUs with just a few lines of code.


# Prerequisites

tf-yarn only supports Python ≥3.6.


# Installation

Note that tf-yarn does not directly depends on the ML frameworks it supports (TensorFlow, torch ...). That way, TensorFlow users don't install torch and conversely by installing tf-yarn. So you must install the ML framework(s) that you use separately (`pip install tensorflow`, `pip install torch` ...).


## Install with Pip

```bash
$ pip install tf-yarn
```


## Install from source

```bash
$ git clone https://github.com/criteo/tf-yarn
$ cd tf-yarn
$ pip install .
```


# TensorFlow prerequisites

Supported versions: [1.15.0 to 2.2.0].

Make sure to have Tensorflow working with HDFS by setting up all the environment variables as described [here](https://docs.w3cub.com/tensorflow~guide/deploy/hadoop).

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


# Quick start

Distributing the training of a model with tf-yarn can be decomposed in two steps:
1. Describe your experiment: write the code that will be executed by the workers involved in the training. This includes the instantiation of the model to train, the training dataset (optionally the validation dataset) and the training loop.
2. Run your experiment: execute your code on yarn.

Refer to the part dedicated to your ML framework (TensorFlow, Pytorch ...) for a detailed description of these two steps


## TensorFlow

tf-yarn supports Keras API and the Estimator API (which was the only high-level API of the first TensorFlow releases).


### Describe your experiment


#### Keras API

A Keras experiment is described by an instance of `tf_yarn.tensorflow.KerasExperiment` composed of the following elements:

- model: compiled Keras model to train
- model_dir: hdfs directory where the model will be checkpointed
- train_params: training parameters that will be provided to `model.fit`. This does not include the training examples (input and target data)
- input_data_fn: function returning the input data (features only) to train the model on
- target_data_fn: function returning the target data (labels only) to train the model on
- validation_data_fn: function returning the data to evaluate the model on

Example:

```python
from tf_yarn.tensorflow import KerasExperiment

def input_data_fn():
    dataset = ...
    return dataset
        .shuffle(1000)
        .batch(128)
        .repeat()

def validation_data_fn():
    dataset = ...
    return dataset
        .shuffle(1000)
        .batch(128)

model = tf.keras.Sequential()
...
opt = tf.keras.optimizers.Adadelta(1.0 * HVD_SIZE)
model.compile(loss='sparse_categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
train_params = {
    "steps_per_epoch": 100,
    "callbacks": my_callbacks
}

experiment = KerasExperiment(
    model=model,
    model_dir=hdfs_dir,
    train_params=train_params,
    input_data_fn=input_data_fn,
    target_data_fn=None,
    validation_data_fn=validation_data_fn
)
```


#### Estimator API

The experiment is described by an instance of `tf_yarn.tensorflow.Experiment` composed of the following elements:
- estimator: the model to train
- train_spec: an instance of [tf.estimator.TrainSpec](https://www.tensorflow.org/api_docs/python/tf/estimator/TrainSpec)
- eval_spec: an instance of [tf.estimator.EvalSpec](https://www.tensorflow.org/api_docs/python/tf/estimator/EvalSpec)

```python
from tf_yarn.tensorflow import Experiment

estimator = tf.estimator.Estimator(model_fn=model_fn)
train_spec = tf.estimator.TrainSpec(input_fn, max_steps=1)
eval_spec = tf.estimator.EvalSpec(input_fn, steps=1)
experiment = Experiment(estimator, train_spec, eval_spec)
```


### Run your experiment

To run your experiment on yarn, simply call the method `tf_yarn.tensorflow.run_on_yarn`. The only mandatory parameter is experiment_fn which must be a function accepting no parameter and returning your object `tf_yarn.tensorflow.KerasExperiment` or `tf_yarn.tensorflow.Experiment` which describes your experiment.

```python
from tf_yarn.tensorflow import run_on_yarn, KerasExperiment

def experiment_fn():
    ...
    return KerasExperiment(
        model=model,
        model_dir=hdfs_dir,
        train_params=train_params,
        input_data_fn=input_data_fn,
        target_data_fn=None,
        validation_data_fn=validation_data_fn
    )

run_on_yarn(
    experiment_fn
)
```

The default distribution strategy is [ParameterServerStrategy](https://www.tensorflow.org/tutorials/distribute/parameter_server_training) which belongs to the group of asynchronous distribution strategies.
Although this distribution strategy works very well with the Estimator API, we did not manage to make it work with the Keras API (with TensorFlow <= 2.2). So we advise Keras users to use [horovod gloo](https://github.com/criteo/tf-yarn/blob/master/docs/HorovodWithGloo.md) for distributing the training. Note that horovod gloo is a synchronous distribution strategy based on all-reduce ops:

```python
from tf_yarn.tensorflow import run_on_yarn, KerasExperiment

def experiment_fn():
    ...
    return KerasExperiment(
        model=model,
        model_dir=hdfs_dir,
        train_params=train_params,
        input_data_fn=input_data_fn,
        target_data_fn=None,
        validation_data_fn=validation_data_fn
    )

run_on_yarn(
    experiment_fn,
    custom_task_module="tf_yarn.tensorflow.tasks.gloo_allred_task"
)
```


## Pytorch


### Describe your experiment

A Pytorch experiment is described by an instance of `tf_yarn.pytorch.PytorchExperiment` composed of the following elements:

- model: model to train
- main_fn: Main function run to train the model. This function is executed by all workers involved in the training. It must accept these inputs: model to train, train dataloader, device (cpu:0, cpu:1, cuda:0, cuda:1 ...) allocated to the worker for the training and rank (worker id).
- Training dataset: training dataset (instance of `torch.utils.data.Dataset`).
- dataloader_args: parameters (batch size, number of workers, collate function ...) passed to the dataloader used to load and iterate over the training dataset. Instance of `tf_yarn.pytorch.DataLoaderArgs`.
- n_workers_per_executor: number of workers per yarn executor.
- tensorboard_hdfs_dir: HDFS directory where tensorboard results will be written at the end of the training
- ddp_args: DistributedDataParallel parameters. Refer to [Pytorch documentation](https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel). Instance of `tf_yarn.pytorch.DistributedDataParallelArgs`

```python
from tf_yarn.pytorch import PytorchExperiment

def main_fn(
    model: torch.nn.Module,
    trainloader: torch.utils.data.dataloader.DataLoader,
    device: str,
    rank: int
):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(10):
        trainloader.sampler.set_epoch(epoch)
        for i, data in enumerate(trainloader, 0):
            data = data.to(rank)
            prediction = model(data)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

experiment = PytorchExperiment(
    model=model,
    main_fn=main_fn,
    train_dataset=trainset,
    dataloader_args=DataLoaderArgs(batch_size=4, num_workers=2),
    n_workers_per_executor=2
)
```


### Run your experiment

To run your experiment on yarn, simply call the method `tf_yarn.pytorch.run_on_yarn`. The only mandatory parameters are:
- experiment_fn: must be a function accepting no parameter and returning your object `tf_yarn.pytorch.PytorchExperiment` which describes your experiment.
- task_specs: describe yarn resources that you want to use for your experiment.

```python
from tf_yarn.pytorch import run_on_yarn, PytorchExperiment, TaskSpec

def experiment_fn():
    ...
    return PytorchExperiment(
        model=model,
        main_fn=main_fn,
        train_dataset=trainset,
        dataloader_args=DataLoaderArgs(batch_size=4, num_workers=2),
        n_workers_per_executor=2
    )

run_on_yarn(
    experiment_fn,
    task_specs={
        "worker": TaskSpec(memory=48*2**10, vcores=48, instances=2, label=NodeLabel.GPU)
    }
)
```

The distribution strategy is [DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) which belongs to the family of synchronous distribution strategies.


# run_on_yarn

The method run_on_yarn exposes several parameters that let you configure the yarn job that will be created to train your model on yarn:

- `pyenv_zip_path`: Path to an archive of your python environment that will be used by executors to run your experiment. It can be a zipped conda env or a pex archive.
If your python environement is different between CPU and GPU, you can provide a dictionnary from `tf_yarn.topologies.NodeLabel` to a python environment. Example:

```python
from tf_yarn import NodeLabel
...
run_on_yarn(
    ...,
    pyenv_zip_path={
        NodeLabel.CPU: "viewfs://root/path/to/env-cpu",
        NodeLabel.GPU: "viewfs://root/path/to/env-gpu"
    }
)
```

If no archive is provided, tf-yarn will automatically package your active python environment in a pex.


- `task_specs`: used to define the resources that you need for your experiment. Dictionary from task names (ps, worker, chief, evaluator, tensorboard ...) to `tf_yarn.topologies.TaskSpec`. Example:

```python
from tf_yarn import TaskSpec, NodeLabel
...
run_on_yarn(
    ...,
    task_specs={
        "worker": TaskSpec(memory=48*2**10, vcores=48, instances=2, label=NodeLabel.GPU),
        "tensorboard": TaskSpec(memory=16*2**10, vcores=4, instances=1, label=NodeLabel.CPU)
    }
)
```

In this example, we are requesting 2 executors with GPUs, 48 vcores and 48 GBs for workers and 1 executor with 4 vcores and 16 GBs for tensorboard.


- `files`: local files or directories to upload on the executors. Dictionary from target location (on executor) to local location (on your local machine). Target locations must be relative to the executor root directory. Note that the executor root directory is appended to ``PYTHONPATH``. Therefore, any listed Python module will be importable.


- `env`: environment variables to set on executors. Dictionary from variable name to variable value. Example:

```python
run_on_yarn(
    ...,
    env={"HADOOP_CONF_DIR": "/etc/hadoop/conf.all"}
)
```


- `queue`: yarn queue to schedule your job in. Example:

```python
run_on_yarn(
    ...,
    queue="ml-gpu"
)
```


- `acls`: configures the application-level Access Control Lists (ACLs). Optional, defaults to ACLs all access. See `ACLs <https://jcrist.github.io/skein/specification.html#acls>` for details.


- `file_systems`: `skein` the library underlying `tf_yarn` automatically acquires a delegation token
for ``fs.defaultFS`` on security-enabled clusters. This should be enough for most
use-cases. However, if your experiment needs to access data on namenodes other than
the default one, you have to explicitly list them in the `file_systems` argument. This would instruct `skein` to acquire a delegation token for
these namenodes in addition to ``fs.defaultFS``:

```python
run_on_yarn(
    ...,
    file_systems=["hdfs://preprod"]
)
```

- `nb_retries`: number of times the yarn application is retried in case of failures

- `name`: Name of the yarn application


# Model evaluation

This feature is not supported for Pytorch.

Model training and model evaluation can be run independently. To do so, you must
use the parameter `custom_task_module` of `run_on_yarn`.

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

# Examples

Please refer to the various examples available in [examples](https://github.com/criteo/tf-yarn/tree/master/examples)


# Other documentations

[MLflow](https://www.mlflow.org/docs/latest/quickstart.html) to track experiments.
More infos [here](https://github.com/criteo/tf-yarn/blob/master/docs/MLflow.md).

[Tensorboard](https://github.com/criteo/tf-yarn/blob/master/docs/Tensorboard.md) can be spawned in a separate container during learnings.

Two alternatives to TensorFlow's distribution strategies are available:
[Horovod with gloo](https://github.com/criteo/tf-yarn/blob/master/docs/HorovodWithGloo.md) and [tf-collective-all-reduce](https://github.com/criteo/tf-collective-all-reduce)
