import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Dataset
trainset = torchvision.datasets.CIFAR10(root="./data/train", train=True, transform=transform, download=True)
testset = torchvision.datasets.CIFAR10(root="./data/test", train=False, transform=transform, download=True)

# DataLoader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, num_workers=2, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, num_workers=2, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# Initializing the Loss Function and the Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(2):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # get inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimizer
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Testing
correct = 0
total = 0
with torch.no_grad(): # If there is no backward call, optimize the memory
    for data in testloader:
        # get input
        inputs, labels = data

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1) # Get the prediction
        total += labels.size(0)
        correct += (predicted == labels).sum().item() # Element wise == and then sum on all values when predicted is the same as the labels
    
print('Accuracy of the network on the 10000 test images. %d %%' % (100*correct/total))