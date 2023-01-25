import torch
import numpy as np

# Tensor Initialization

data = [[1,2], [3,4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Ones Tensor: \n {x_ones} \n")
print(f"Random Tensor: \n {x_rand} \n")

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

# Tensor Attributes

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device of tensor: {tensor.device}")

# Tensor Operations

if torch.backends.mps.is_available():
    tensor = tensor.to('mps')
    print(f"Device of tensor: {tensor.device}")

tensor = torch.ones(4,4)
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor,tensor,tensor], dim=1)
print(t1)

print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
print(f"tensor * tensor \n {tensor * tensor} \n")

print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
print(f"tensor @ tensor.T \n {tensor @ tensor.T} \n")

tensor.add_(5)
print(tensor)

# Bridge with Numpy

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")






