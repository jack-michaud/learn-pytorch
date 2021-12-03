# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

# autograd is pytorch's automatic differentiation engine.
# autograd takes care of forward and back propagation.

import torch, torchvision

model = torchvision.models.resnet18(pretrained=True)

data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# Get the loss from the prediction
prediction = model(data)
loss = (prediction - labels).sum()
# Backpropoagates the loss through the network. The prediction stores the gradients in `.grad`.
loss.backward()

# Creating an SGD optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# The optimizer will update the weights and biases from the stored gradient
optimizer.step()
