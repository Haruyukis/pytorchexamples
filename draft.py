import torch

x = torch.rand(3, requires_grad=True)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)

x.backward(v)
print(x.grad)