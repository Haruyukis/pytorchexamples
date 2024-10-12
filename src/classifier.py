"""
- For images, packages such as Pillow, OpenCV are useful
- For audio, packages such as scipy and librosa
- For text, either raw Python, or Cython based loading, or NLTK and SpaCy are useful

For vision, look inside **torchvision**, containing common datasets such as ImageNet, CIFAR10, MNIST, etc. And data transformers for images via torchvision.dataset and torch.utils.data.DataLoader
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

## Training an image classfier

# 1. Load and normalize CIFAR10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

training_set = torchvision.datasets.CIFAR10(root="./data/CIFAR10/training", train=True, transform=transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./data/CIFAR10/test", train=False, transform=transform, download=True)

training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Displaying some images
# import matplotlib.pyplot as plt
# import numpy as np

# # functions to show an image


# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# # get some random training images
# dataiter = iter(training_loader)
# images, labels = next(dataiter)

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# 3. Define a CNN with 3-channels...

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        out = self.pool(F.relu(self.conv1(input)))
        out = self.pool(F.relu(self.conv2(out)))

        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
net = Net()

# 4. Define a Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# 5. Train the network
# for epoch in range(2):

#     running_loss = 0.0

#     for i, data in enumerate(training_loader, 0):

#         # Zero-Grad
#         optimizer.zero_grad()

#         # Forward
#         inputs, labels = data
#         outputs = net(inputs)

#         # Backward
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss
#         if i%2000 == 1999:
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0

# print("Finished Training")

# # 6. Save & Load the model

PATH = "./model/cifar_net.pth"
# torch.save(net.state_dict(), PATH)

net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

# 7. Accuracy
correct = 0
total = 0

with torch.no_grad():    
    for data in test_loader:

        inputs, labels = data

        # Forward
        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1) # Maximum value given to the dim 1. The value are tuple with logits and labels

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in test_loader:

        inputs, labels = data

        # Forward
        outputs = net(inputs)

        _, predictions = torch.max(outputs, 1)

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] +=1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')