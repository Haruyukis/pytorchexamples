import torch
from torchvision.models import resnet18, ResNet18_Weights

## A Gentle Introduction to **torch.autograd**

model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# Forward pass
prediction = model(data)

loss = (prediction - labels).sum()
loss.backward()

optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optim.step()

## Differentiation in Autograd
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

# The call of Q.backward() will put the value to each variables that was usefull to compute Q
print(9*a**2 == a.grad) 
print(-2*b == b.grad)

# Exclusion from the DAG
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients?: {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")

# Frozen parameters - Freeze all the parameters in the network.
model = resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(512, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer.step() # Only the model.fc will compute the gradient descent, because other parameters are frozen.
