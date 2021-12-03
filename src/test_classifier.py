import torch
from src.classifier import Net
from src.train_classifier import testloader, classes

dataiter = iter(testloader)
images, labels = dataiter.next()

net = Net()
net.load_state_dict(torch.load("./cifar_net.pth"))
net.eval()

print("Ground truth labels: " + ", ".join([classes[labels[i]] for i in range(4)]))

predictions = []
outputs = net(images[:4])
outputs = torch.max(outputs, 1)
print("Predicted labels: " + ", ".join([classes[outputs.indices[i]] for i in range(4)]))
