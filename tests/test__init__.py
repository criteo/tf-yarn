from unittest import mock
from tf_yarn import (
    NodeLabel,
    _make_conda_envs
)


@mock.patch("tf_yarn.create_and_pack_conda_env")
def test_make_conda(mock_packer):
    mock_packer.return_value = "env.zip"
    res = _make_conda_envs("python3.6", ["awesomepkg"])
    assert res.keys() == {NodeLabel.CPU, NodeLabel.GPU}
    assert res[NodeLabel.CPU] == "env.zip"
    assert res[NodeLabel.GPU] == "env.zip"
