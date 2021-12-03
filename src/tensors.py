# https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#tensors
import torch
import numpy as np

data = [[1, 2], [3, 4]]

# You can create a tensor from a list
x_data = torch.tensor(data)

# Or you can create it from a numpy array too!
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

assert x_data.equal(x_np)

# You can create tensors from other tensors
# The _like functions assume the same properties as the
# source tensor. You can override them through the kwargs.
x_ones = torch.ones_like(x_data)
# Equivalent: torch.ones(x_data.shape)
print(f"Ones: {x_ones}")
x_rand = torch.rand_like(x_data, dtype=torch.float)
# Equivalent: torch.rand(x_data.shape)
print(f"Random: {x_rand}")

# You can easily access the datatype, shape, and device its stored on.
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# There are tensor operations described here
# https://pytorch.org/docs/stable/torch.html

# Moving the tensor to the GPU
if torch.cuda.is_available():
    x_data = x_data.to("cuda")

# You get numpy array slicing too:
tensor = torch.ones(4, 4)
tensor[:, 1] = 0
tensor[:, 3] = 0
tensor[1, :] = 1
tensor[3, :] = 1
print(tensor)

# You can concatenate tensors along a dimension.
print(torch.cat([tensor, tensor], dim=1))
print(torch.cat([tensor, tensor], dim=0))

# Element-wise multiplication:
print(tensor * tensor)
# Matrix multiplication
print(tensor.matmul(tensor.T))
print(tensor @ tensor.T)

# The _ suffix means an operation is in place
tensor = torch.ones(4, 4)
tensor[:, 1] = 0
tensor.add_(5)
print(tensor)
