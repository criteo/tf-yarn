from typing import Optional
from cluster_pack.packaging import PythonEnvDescription
from tf_yarn import topologies


INDEPENDENT_WORKERS_MODULE = "tf_yarn.tensorflow.tasks._independent_workers_task"
TENSORBOARD_MODULE = "tf_yarn.tensorflow.tasks._tensorboard_task"


def gen_task_cmd(pyenv: PythonEnvDescription,
                 task_type: str,
                 custom_task_module: Optional[str]) -> str:

    if task_type == "tensorboard":
        containers_module = TENSORBOARD_MODULE
    elif task_type in topologies.ALL_TASK_TYPES:
        if custom_task_module:
            containers_module = custom_task_module
        else:
            containers_module = INDEPENDENT_WORKERS_MODULE
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    return f"{pyenv.interpreter_cmd} -m {containers_module} "
