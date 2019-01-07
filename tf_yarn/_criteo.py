import os


def is_criteo():
    return "CRITEO_ENV" in os.environ


def get_requirements_file():
    return "requirements.txt"
