"""
The tests in this module should be ran using the ``hadoop-test-cluster``
package::

     $ htcluster startup --mount .:tf-yarn
     $ htclsuter login
     $ cd tf-yarn
     $ pip install -r requirements.txt
     $ pip install pytest
     $ pytest -vv tests/test_htcluster.py
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
def test_single_chief_no_hdfs_access():
    root_dir = os.path.dirname(here)
    output = check_output(
        [sys.executable, os.path.join(here, "id_estimator_experiment.py")],
        stderr=PIPE,
        env={**os.environ, "PYTHONPATH": root_dir})
    assert output and b"FAILED" not in output
