import os
from subprocess import check_output


# The following is specific to the Criteo infra. Moreover, the
# definitions assume that the system submitting the application
# is "sufficiently close" to that of the containers.


def hdfs():
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


def cuda():
    """TODO"""
    return {}
