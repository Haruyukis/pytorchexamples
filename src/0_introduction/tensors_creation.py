import torch

# Creating your first tensor
z = torch.zeros(5, 3)

print(z)
print(z.dtype)

# Filling the tensor with 1 and using int.
i = torch.ones((5, 3), dtype=torch.int16)
print(i)

# Creating a random tensor, Important: need to initialize weights randomly.
torch.manual_seed(1729)
r1 = torch.rand(2, 2)
print('A random tensor:')
print(r1)

r2 = torch.rand(2, 2)
print('\nA different random tensor:')
print(r2) # new values

torch.manual_seed(1729)
r3 = torch.rand(2, 2)
print('\nShould match r1:')
print(r3) # repeats values of r1 because of re-seed