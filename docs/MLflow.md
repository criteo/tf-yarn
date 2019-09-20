# MLflow

MLflow tracking is activated when mlflow package is installed (`pip install mlflow` or `conda install mlflow`) and when the tracking uri is set.

To setup (MLflow tracking)[(https://www.mlflow.org/docs/latest/tracking.html#where-runs-are-recorde] you need set the MLFLOW_TRACKING_URI environment variable to a tracking server’s URI or call [`mlflow.set_tracking_uri()`](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri).

The MLflow example can be found here [here](https://github.com/criteo/tf-yarn/blob/master/examples/mlflow_example.py).

Currently tf-yarn logs the following metrics by default:
- Container duration times in seconds
- Container log urls & final status
- Learning speed of the chief only (steps/sec)
- Statistic about the evaluator (Awake/idle ratio, Eval step mean duration in seconds)

Distributed metrics are implemented via TensorFlow hooks. You can add you own metrics by adding new Hooks and choose if you want to log from all nodes or from specific nodes (chief, worker, ..) via the tf_yarn.cluster module.

For example to log a metric from the evaluator only you can call:
```
from tf_yarn import cluster, mlflow

class MyHook(tf.train.SessionRunHook):
   ...

   def after_run(self, run_context, run_values):
       if cluster.is_evaluator():
           mlflow.log_tag(..)
```
An example hook logging the steps/sec can be found here [StepPerSecondHook](https://github.com/criteo/tf-yarn/blob/master/tf_yarn/metrics.py#L66)
