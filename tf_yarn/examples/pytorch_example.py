import os
import getpass
from uuid import uuid4

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


from tf_yarn.pytorch import run_on_yarn, TaskSpec, NodeLabel
from tf_yarn.pytorch import PytorchExperiment, DataLoaderArgs
from tf_yarn.pytorch import model_ckpt


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main_fn(
    model: torch.nn.Module,
    trainloader: torch.utils.data.dataloader.DataLoader,
    device: str,
    rank: int,
    tb_writer: torch.utils.tensorboard.writer.SummaryWriter
):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model_ckpt_path = f"viewfs://root/user/{getpass.getuser()}/toy_model_ckpt_{str(uuid4())}"
    print(f"Will checkpoint model in {model_ckpt_path}")

    train_ite_num = 0
    running_loss = 0.0
    running_ite = 0
    for epoch in range(10):
        trainloader.sampler.set_epoch(epoch)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_ite += 1
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{os.getpid()}] [{epoch + 1}, {i + 1}] loss: {running_loss / 2000}')
                tb_writer.add_scalar(
                    'training_loss_sum', running_loss / running_ite, train_ite_num
                )
                running_loss = 0.0
                running_ite = 0
            train_ite_num += 1
        model_ckpt.save_ckpt(model_ckpt_path, model, optimizer, epoch)

    print('Finished Training')


def experiment_fn():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    return PytorchExperiment(
        model=Net(),
        main_fn=main_fn,
        train_dataset=trainset,
        dataloader_args=DataLoaderArgs(batch_size=4, num_workers=2),
        n_workers_per_executor=2
    )


if __name__ == "__main__":
    run_on_yarn(
        experiment_fn=experiment_fn,
        task_specs={
            "worker": TaskSpec(memory=48*2**10, vcores=48, instances=2, label=NodeLabel.GPU)
        },
        queue="ml-gpu"
    )
