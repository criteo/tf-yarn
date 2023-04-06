import os
import torch
import torchvision

import torch.distributed as dist

import torch.nn as nn
import torch.nn.functional as F

from tf_yarn.distributed.task import get_task
from tf_yarn.topologies import TaskSpec, NodeLabel
from tf_yarn import client
import cluster_pack


def train_fcn():
    local_rank = 0  # TBD later
    _, rank, size, master_addr, master_port, _ = get_task()
    print(f'master: {master_addr}:{master_port}')

    # Initialization of Pytorch distributed bakend
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ["PYTORCH_DPP_RANK"] = str(rank)
    dist.init_process_group(backend='nccl', world_size=size, rank=rank)

    nb_epoch = 50

    # ######### Training
    # Network def
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    # Set of exclusive GPU for the process
    gpu = torch.device(f"cuda:{local_rank}")

    # Transfer the network to GPU + prepare it for distributed training (incl. weight broadcast)
    model = Net().to(gpu)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Optimization params
    batch_size = 128
    batch_size_per_gpu = batch_size // size
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), 1e-4)

    # Data loading + distributed sampler
    train_dataset = torchvision.datasets.MNIST(
        f'./tmp/mnist_{local_rank}', transform=torchvision.transforms.ToTensor(), download=True
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size_per_gpu, shuffle=False, num_workers=4,
        pin_memory=True, sampler=train_sampler
    )

    # Training loop
    for epoch in range(nb_epoch):
        train_sampler.set_epoch(epoch)
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(gpu, non_blocking=True)
            labels = labels.to(gpu, non_blocking=True)
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            if rank == 0 and i % 100 == 0:
                print(f'Train epoch: {epoch} [{i* len(images)}/{len(train_loader.dataset)}]'
                      f'\tLoss: {loss.item()}')
        # Some operations like model saving only on one process
        if rank == 0:
            print('saving model dict at ./tmp/model.pt')
            torch.save(ddp_model.state_dict(), './tmp/model.pt')


if __name__ == "__main__":
    print("packaging venv")
    pyenv_path, _ = cluster_pack.upload_env(allow_large_pex=True)
    print(f"venv uploaded to {pyenv_path}")

    task_specs = {
        "worker": TaskSpec(memory=48 * 2 ** 10, vcores=48, instances=2, label=NodeLabel.GPU)
    }

    client.run_on_yarn(
        lambda: train_fcn,
        task_specs,
        custom_task_module="tf_yarn.distributed.task",
        queue="ml-gpu",
        pyenv_zip_path=pyenv_path
    )
