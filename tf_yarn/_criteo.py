import os


def is_criteo():
    return "CRITEO_ENV" or "TARGET_ENVIRONMENT" in os.environ


def get_requirements_file():
    if is_criteo():
        return "criteo.requirements.txt"
    else:
        return "requirements.txt"
