import os
from subprocess import check_output

from tf_skein import TaskFlavor

# The following is specific to the Criteo infra. Moreover, the
# definitions assume that the system submitting the application
# is "sufficiently close" to that of the containers.


def hdfs_vars():
    """TODO"""
    hadoop_home = os.environ.setdefault("HADOOP_HOME", "/usr/lib/hadoop")
    hadoop_classpath = check_output([
        os.path.join(hadoop_home, "bin", "hadoop"),
        "classpath",
        "--glob"
    ])
    return {
        "HADOOP_HDFS_HOME": "/usr/lib/hadoop-hdfs",
        "CLASSPATH": hadoop_classpath.decode().strip(),
        "LD_LIBRARY_PATH": ":".join([
            f"{os.environ['JAVA_HOME']}/jre/lib/amd64/server",
            "/usr/lib/hadoop-criteo/hadoop/lib/native"
        ])
    }


def node_label_fn(task_flavor: TaskFlavor) -> str:
    if task_flavor is TaskFlavor.CPU:
        return ""
    else:
        assert task_flavor is TaskFlavor.GPU
        return "gpu"
