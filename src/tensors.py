import torch
import numpy as np

## Tensor initialization

# Directly from data
with torch.no_grad():
    x = [[1, 2], [2, 3]]
    x_data = torch.tensor(x)
    print(x_data)

    # Directly from numpy array
    x_numpy = np.array(x)
    tensor_from_numpy = torch.from_numpy(x_numpy)
    print(tensor_from_numpy)

    # From another tensor to keep the shape and dtype
    x_ones = torch.ones_like(x_data) 
    print(f"Ones tensor: \n {x_ones} \n")

    x_rand = torch.rand_like(x_data, dtype=torch.float) # Overrides the dtype
    print(f"Random Tensor: \n {x_rand} \n")

    # With random or constant values
    shape = (2, 3)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")

# Tensor Attributes
with torch.no_grad():
    tensor = torch.rand(3, 4)

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")

    if torch.cuda.is_available():
        tensor = tensor.to("cuda")
        print(f"Device tensor is stored on: {tensor.device}")

## Tensor Operations
with torch.no_grad():
    tensor = torch.ones(4, 4)
    if torch.cuda.is_available():
        tensor = tensor.to("cuda")
        print(f"Device tensor is stored on: {tensor.device}")

    # Standard numpy-like indexing and slicing
    tensor[:, 1] = 0 # Modify the second column to 0 [Rows, Columns]
    print(tensor)

    # Joining tensors
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)
    tensor.mul_(2)
    # Multiplying tensors
    # This computes the element-wise product
    print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
    # Alternative syntax:
    print(f"tensor * tensor \n {tensor * tensor }")

    print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
    # Alternative syntax:
    print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

    # In-place operations - Operations that have a _ suffix are in-place.
    # In-place operations save some memory but affect autograd since the object itself is modifying through the hidden layer.
    print(tensor, "\n")
    tensor.add_(5)
    print(tensor)

## Bridge with NumPy
with torch.no_grad():
    # Tensor to NumPy array
    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()
    print(f"n: {n}")
    
    # A change in the tensor reflects in the NumPy array
    t.add_(1)
    print(f"t: {t}")
    print(f"n: {n}")

    # NumPy array to Tensor
    n = np.ones(5)
    t = torch.from_numpy(n)

    # Changes in the NumPy array reflects in the tensor
    np.add(n, 1, out=n) # In-place operation in NumPy
    print(f"t: {t}")
    print(f"n: {n}")






    


