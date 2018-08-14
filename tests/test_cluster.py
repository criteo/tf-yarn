"""
The tests in this module should be ran using the ``hadoop-test-cluster``
package::

     $ htcluster startup --mount .:tf-skein
     $ htclsuter login
     $ cd tf-skein
     $ pip install -r requirements.txt
     $ pip install pytest
     $ pytest -vv tests/test_cluster.py
"""

import os
import sys
from subprocess import check_output, CalledProcessError, PIPE

import pytest

here = os.path.dirname(__file__)


is_hadoop = False
try:
    check_output("hadoop version".split(), shell=True)
except CalledProcessError:
    pass
else:
    is_hadoop = True


@pytest.mark.skipif(
    not is_hadoop,
    reason="not running on a Hadoop edge node/gateway"
)
def test_end_to_end():
    root_dir = os.path.dirname(here)
    examples_dir = os.path.join(root_dir, "examples")
    hadoop_classpath = check_output(
        "hadoop classpath --glob".split()).strip().decode()
    output = check_output(
        [sys.executable, os.path.join(examples_dir, "cpu_example.py")],
        stderr=PIPE,
        env={
            **os.environ,
            "CLASSPATH": hadoop_classpath,
            "LD_LIBRARY_PATH": f"{os.environ['JAVA_HOME']}/lib/amd64/server",
            "PYTHONPATH": root_dir + ":" + examples_dir,
        })
    assert b"SUCCEEDED" in output
