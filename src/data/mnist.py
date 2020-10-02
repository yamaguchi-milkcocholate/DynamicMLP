from __future__ import annotations
from torch.utils.data import random_split, DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms


def load_mnist(batch_size=64, core_num=2):
    train_dataset = dsets.MNIST(
        root='./data_mnist/',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    test_dataset = dsets.MNIST(
        root='./data_mnist/',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    train_dataset, valid_dataset = random_split(
        train_dataset,
        [48000, 12000]
    )
    print(f"train = {len(train_dataset)}\nvalid = {len(valid_dataset)}\n test = {len(test_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=core_num
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=core_num
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=core_num
    )

    return train_loader, valid_loader, test_loader
