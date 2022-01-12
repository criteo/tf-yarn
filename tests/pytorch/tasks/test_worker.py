import os
from tf_yarn.pytorch.experiment import DataLoaderArgs

import mock
import pytest
import torch

from tf_yarn.pytorch.tasks import worker


MODULE_UNDER_TEST = "tf_yarn.pytorch.tasks.worker"


def test_get_device():
    with mock.patch(f"{MODULE_UNDER_TEST}.torch") as torch_mock:
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.device_count.return_value = 2
        assert worker._get_device(0) == 0
        assert worker._get_device(1) == 1
        assert worker._get_device(2) == 0
        assert worker._get_device(3) == 1


def test_setup_master_rank_0():
    with mock.patch(f"{MODULE_UNDER_TEST}._internal") as internal_mock:
        addr = "127.0.0.1"
        port = 1313
        internal_mock.reserve_sock_addr.return_value.__enter__.return_value = (addr, port)
        kv_store = dict()
        skein_client = mock.Mock()
        skein_client.kv = kv_store
        worker._setup_master(skein_client, 0)
        assert kv_store.get(worker.MASTER_ADDR, None) == addr.encode()
        assert kv_store.get(worker.MASTER_PORT, None) == str(port).encode()
        assert os.environ.get(worker.MASTER_ADDR, None) == addr
        assert os.environ.get(worker.MASTER_PORT, None) == str(port)


def test_setup_master():
    addr = "127.0.0.1"
    port = str(1313)
    kv_store = {
        worker.MASTER_ADDR: addr.encode(),
        worker.MASTER_PORT: port.encode()
    }
    skein_client = mock.Mock()
    skein_client.kv.wait = kv_store.get
    worker._setup_master(skein_client, 1)
    assert os.environ.get(worker.MASTER_ADDR, None) == addr
    assert os.environ.get(worker.MASTER_PORT, None) == port


@pytest.mark.parametrize("shuffle", [(True,), (False,)])
def test_create_dataloader(shuffle):
    class _FakeDataset(torch.utils.data.Dataset):
        def __getitem__(self, _: int) -> None:
            pass

        def __len__(self) -> int:
            return 0

    dataloader_args = DataLoaderArgs(batch_size=200, shuffle=shuffle)
    with mock.patch(f"{MODULE_UNDER_TEST}.DistributedSampler") as sampler_mock, \
            mock.patch(f"{MODULE_UNDER_TEST}.torch.utils.data.DataLoader") as dataloader_mock:
        dataset = _FakeDataset()
        worker._create_dataloader(dataset, dataloader_args)
        sampler_mock.call_args_list == [mock.call(dataset, shuffle=shuffle)]
        dataloader_mock.call_args_list == [
            mock.call(
                dataset, sampler=sampler_mock, batch_size=200, shuffle=False, num_workers=0,
                pin_memory=False, drop_last=False, timeout=0, prefetch_factor=2
            )
        ]
