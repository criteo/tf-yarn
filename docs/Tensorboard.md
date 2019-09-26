# Tensorboard

You can use Tensorboard with TF Yarn.
Tensorboard is automatically spawned in a separate container on YARN when using a default `task_specs`.
If you use a custom `task_specs`, you must add explicitly a Tensorboard task to your configuration.

```python
run_on_yarn(
    ...,
    task_specs={
        "chief": TaskSpec(memory="2 GiB", vcores=4),
        "worker": TaskSpec(memory="2 GiB", vcores=4, instances=8),
        "ps": TaskSpec(memory="2 GiB", vcores=8),
        "evaluator": TaskSpec(memory="2 GiB", vcores=1),
        "tensorboard": TaskSpec(memory="2 GiB",
                                vcores=1,
                                tb_termination_timeout_seconds=30,
                                tb_model_dir=model_dir,
                                tb_extra_args=None)
    }
)
```

Optional parameters:
* tb_termination_timeout_seconds: controls how many seconds each tensorboard instance must stay alive after the end of the run. Defaults to 30 seconds
* tb_model_dir: to configure a model directory. If None it will extract the model_dir from the estimator's `run_config`. It is always better to specifiy the model_dir as we don't need to evaluate the experiment_fn and tehrefore tensorboard wil lstartup faster
* tb_extra_args: appends command line arguments to the mandatory ones (--logdir and --port). Defaults to None

The full access URL of each tensorboard instance is advertised as a `url_event` starting with "Tensorboard is listening at...".
Typically, you will see it appearing on the standard output of a `run_on_yarn` call.

