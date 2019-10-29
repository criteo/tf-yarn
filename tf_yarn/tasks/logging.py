
import logging.config
import uuid
import os


def setup():
    # tensorflow imports in tf_yarn.__init__ already have set up some loggers
    # erase them with a clean config
    log_conf_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "default.log.conf")
    logging.config.fileConfig(log_conf_file)
