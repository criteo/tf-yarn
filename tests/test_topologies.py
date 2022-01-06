import pytest

from tf_yarn import topologies
from tf_yarn.topologies import MAX_MEMORY_CONTAINER, MAX_VCORES_CONTAINER


def test_single_server_topology():
    with pytest.raises(ValueError):
        topologies.single_server_topology(memory=MAX_MEMORY_CONTAINER + 1)
    with pytest.raises(ValueError):
        topologies.single_server_topology(vcores=MAX_VCORES_CONTAINER + 1)


def test_ps_strategy_topology():
    with pytest.raises(ValueError):
        topologies.ps_strategy_topology(memory=MAX_MEMORY_CONTAINER + 1)
    with pytest.raises(ValueError):
        topologies.ps_strategy_topology(vcores=MAX_VCORES_CONTAINER + 1)
    with pytest.raises(ValueError):
        topologies.ps_strategy_topology(nb_ps=0)
