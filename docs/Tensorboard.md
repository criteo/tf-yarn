# Tensorboard

You can use Tensorboard with TF Yarn.
Tensorboard is automatically spawned when using a default task_specs. Thus running as a separate container on YARN.
If you use a custom task_specs, you must add explicitly a Tensorboard task to your configuration.

```python
run_on_yarn(
    ...,
    task_specs={
        "chief": TaskSpec(memory="2 GiB", vcores=4),
        "worker": TaskSpec(memory="2 GiB", vcores=4, instances=8),
        "ps": TaskSpec(memory="2 GiB", vcores=8),
        "evaluator": TaskSpec(memory="2 GiB", vcores=1),
        "tensorboard": TaskSpec(memory="2 GiB", vcores=1, instances=1, termination_timeout_seconds=30)
    }
)
```

Both instances and termination_timeout_seconds are optional parameters.
* instances: controls the number of Tensorboard instances to spawn. Defaults to 1
* termination_timeout_seconds: controls how many seconds each tensorboard instance must stay alive after the end of the run. Defaults to 30 seconds

The full access URL of each tensorboard instance is advertised as a _url_event_ starting with "Tensorboard is listening at...".
Typically, you will see it appearing on the standard output of a _run_on_yarn_ call.

### Environment variables
The following optional environment variables can be passed to the tensorboard task:
* TF_BOARD_MODEL_DIR: to configure a model directory. Note that the experiment model dir, if specified, has higher priority. Defaults: None
* TF_BOARD_EXTRA_ARGS: appends command line arguments to the mandatory ones (--logdir and --port): defaults: None