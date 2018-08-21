import os
from subprocess import check_output


def is_criteo():
    return "CRITEO_ENV" in os.environ


def get_default_env():
    if not is_criteo():
        return {}

    # The following is specific to the Criteo infra. Moreover, the
    # definitions assume that the system submitting the application
    # is "sufficiently close" to that of the containers.
    hadoop_home = os.environ.setdefault("HADOOP_HOME", "/usr/lib/hadoop")
    hadoop_classpath = check_output([
        os.path.join(hadoop_home, "bin", "hadoop"),
        "classpath",
        "--glob"
    ])
    return {
        "CLASSPATH": hadoop_classpath.decode().strip(),
        "LD_LIBRARY_PATH": ":".join([
            f"{os.environ['JAVA_HOME']}/jre/lib/amd64/server",
            "/usr/lib/hadoop-criteo/hadoop/lib/native"
        ])
    }
