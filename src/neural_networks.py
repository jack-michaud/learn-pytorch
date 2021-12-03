# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

# The torch.nn package is used for making neural networks
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    # all models should be subclassed by nn.Module

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        The forward propagation function.
        The back propagation is done automatically
        """
        # max pool over a 2x2 window (if it's a square, you can use 1 number)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

# The learnable parameters are found in
parameters = list(net.parameters())
print(len(parameters))
print(parameters[0].size())

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# Zero the gradient buffers and back propagate with random gradients
net.zero_grad()
out.backward(torch.randn(1, 10))

# Compute the loss using a loss fnuction
# A loss function takes an output and a target and computes
# a value that estimates how far away the output is from the target.
# Loss Functions: https://pytorch.org/docs/nn.html#loss-functions

output = net(input)
target = torch.randn(10)
# Shapes the target into the same shape as the output...
# the output is a batch, so this wraps the output in one layer.
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

# Backprop!
net.zero_grad()
print(net.conv1.bias.grad)
loss.backward()
print(net.conv1.bias.grad)
# then update the weights
learning_rate = 0.01
# Manually implementing SGD... W = W - (lr * grad)
# for f in net.parameters():
#    f.data.sub_(learning_rate * f.grad.data)

optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# Zero the gradient buffers in each training loop
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
