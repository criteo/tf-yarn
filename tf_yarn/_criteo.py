import os


def is_criteo():
    return "CRITEO_ENV" in os.environ


def get_requirements_file():
    if is_criteo():
        return "criteo.requirements.txt"
    else:
        return "requirements.txt"


def get_default_env():
    if not is_criteo():
        return {}

    # The following is specific to the Criteo infra. Moreover, the
    # definitions assume that the system submitting the application
    # is "sufficiently close" to that of the containers.
    return {
        "LD_LIBRARY_PATH": ":".join([
            f"{os.environ['JAVA_HOME']}/jre/lib/amd64/server",
            "/usr/lib/hadoop-criteo/hadoop/lib/native"
        ])
    }
