import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 6 input image channels, 16 output channels, 5x5 square convolution
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully-Connected layer - Affine, Linear Linear
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 16 for the output channel of conv2, output of max_pool2d of conv2, with image of size 32
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        c1 = F.relu(self.conv1(input))
        s2 = F.max_pool2d(c1, (2, 2))

        c3 = F.relu(self.conv2(s2))
        s4 = F.max_pool2d(c3, (2, 2)) # Return a tensor of size (N, 16, 5, 5)
        # Flatten operation: purely functional, outputs a (N, 400) tensor
        s4 = torch.flatten(s4, 1)
        # Fully connected layer F5: (N, 400), output a (N, 120)
        f5 = F.relu(self.fc1(s4))
        # Fully connected layer F6: (N, 120), output a (N, 84)
        f6 = F.relu(self.fc2(f5))
        # Fully connected layer output: (N, 84), output a (N, 10)
        output = self.fc3(f6)
        return output

net = Net()
print(net)

# Backward Draft
params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1's .weight

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# net.zero_grad() # important because the gradient associate to each parameters might stacked up
# out.backward(torch.rand(1, 10)) # 1 batch, 10 nodes in the output vector

## Loss Function
output = net(input)
label = torch.rand(10)
label = label.view(1, -1) # make it the same shape as output

# Define the loss
criterion = nn.MSELoss()

# Compute the loss
loss = criterion(output, label)
print("loss value:", loss)

# Backward DAG
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

## Backpropagation
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

lr = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * lr)

## Backpropagation with Optimizer

output = net(input)

# Define the Loss and the Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Zero-Grad
optimizer.zero_grad()

# Compute the backpropagation
loss = criterion(output, label)
loss.backward()
optimizer.step()