import numpy as np
import torch

arr = [[1, 2], [3, 4]]
print(arr)

# convert array to numpy array
np.array(arr)

# convert to PyTorch Tensor
torch.Tensor(arr)

# create default value matrices
np.ones((2, 2))

torch.ones((2, 2))

np.random.rand(2, 2)

torch.rand(2, 2)

# seeds for reproducibility

np.random.seed(0)
np.random.rand(2, 2)

# torch type seed
torch.manual_seed(0)
torch.rand(2, 2)

# numpy and torch bridge

np_array = np.ones((2, 2))
print(np_array)

print(type(np_array))

torch_tensor = torch.from_numpy(np_array)

print(torch_tensor)
print(type(torch_tensor))

# DATA TYPES MATTER
# bridge supports:
# double
# float
# int64, int32, uint8
np_array_new = np.ones((2, 2), dtype=np.int64)
torch.from_numpy(np_array_new)

np_array_new = np.ones((2, 2), dtype=np.int32)
torch.from_numpy(np_array_new)

np_array_new = np.ones((2, 2), dtype=np.uint8)
torch.from_numpy(np_array_new)

np_array_new = np.ones((2, 2), dtype=np.double)
torch.from_numpy(np_array_new)

np_array_new = np.ones((2, 2), dtype=np.float32)
torch.from_numpy(np_array_new)

# tensor type bug summary
# NumPy Array Type	Torch Tensor Type
# int64	            LongTensor
# int32	            IntegerTensor
# uint8	            ByteTensor
# float64	        DoubleTensor
# float32	        FloatTensor
# double	        DoubleTensor

# tensor defaults to float in this case
torch_tensor = torch.ones(2, 2)
type(torch_tensor)

torch_to_numpy = torch_tensor.numpy()
type(torch_to_numpy)

# TENSOR OPERATIONS
# Resizing tensors
a = torch.ones(2, 2)
print(a)
print(a.size())

# this resizes tensor to 4x1
a.view(4)

# this gets the size of the resized tensor
a.view(4).size()

# element-wise addition
a = torch.ones(2, 2)
print(a)

b = torch.ones(2, 2)
print(b)

c = a + b

# alternative
c = torch.add(a, b)

print('Old c tensor')
print(c)

c.add_(a)

print('-'*60)
print('New c tnesor')
print(c)

# Element-wise Subtraction
a - b

# second method
# this method does not change the value of a
print(a.sub(b))
print(a)

# third method
# this method does change the value of a
print(a)
print(a.sub_(b))

# Element wise multiplication
a * b

# second method
print(torch.mul(a, b))
print(a)

# third method
print(a.mul_(b))
print(a)

# Create a tensor filled with ones and zeros
a = torch.ones(2, 2)
print(a)
b = torch.zeros(2, 2)
print(b)

# Element wise division
b / a

# second method
torch.div(b, a)

# third method
b.div_(a)

# creating a tensor of size 10 filled from 1 to 10
a = torch.Tensor([1, 2, 3,  4, 5, 6, 7, 8, 9, 10])
print(a.size())
# size of a is 10 not 10x1

# Tensor arithmatic mean
print(a.mean(dim=0))

a = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

print(a.size())  # size of a is 2x10
# Tensor arithmatic mean dimensions 0 is verticle mean
print(a.mean(dim=0))
print(a.mean(dim=1))
