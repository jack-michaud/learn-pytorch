# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# torchvision has data loaders for common datasets (ImageNet, CIFAR10, MNIST)
# it also has data transformers for images.
from torch import nn, optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torch.utils.data.dataloader as dataloader

# Time to load CIFAR10.
# > Torchvision datasets are PILImage images of range [0, 1]
# > Normalized them into tensors between [-1, 1]
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(  # compose multiple transforms
    [
        # converts PILImage or nparray
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
batch_size = 4
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = dataloader.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = dataloader.DataLoader(
    testset, batch_size=batch_size, shuffle=True, num_workers=2
)
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

from .classifier import Net


def run_train():
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"loss: {running_loss / 2000}")
                running_loss = 0

    torch.save(net.state_dict(), "./cifar_net.pth")
    # State dict:
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html


if __name__ == "__main__":
    run_train()
