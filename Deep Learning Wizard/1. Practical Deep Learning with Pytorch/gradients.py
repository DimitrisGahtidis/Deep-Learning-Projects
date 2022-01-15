import torch

# create tensor with gradients
a = torch.ones((2, 2), requires_grad=True)
print(a)
print(a.requires_grad)  # This should come out true

# second method (sequential)
# create the tensor
a = torch.ones((2, 2))

# add gradient requirement
a.requires_grad_()

# check if gradient requirement has been applied
print(a.requires_grad)

# addition with gradients
b = torch.ones((2, 2), requires_grad=True)
print(a + b)
print(torch.add(a, b))

# multiplication with gradients
print(a * b)
print(torch.mul(a, b))

# WHAT EXACTLY IS requires_grad? - Allows calculation of gradients
# with respect to the tensor that all allows gradients accumulartion
# example y_i = 5(x_i + 1)^2

x = torch.ones(2, requires_grad=True)
print(x)

y = 5 * (x + 1) ** 2
print(y)

o = (1/2) * torch.sum(y)
print(o)

# CALCULATE THE FIRST DERIVATIVE
# Recaping:
# y_i = 5(x_i + 1)^2
# o = 1/2 sum y_i
# o = 1/2 sum 5(x_i + 1)^2
# implying that do/dx_i = (1/2)(10(x_i + 1))

# get the first derivative
o.backward()

# print the first derivative
print(x.grad)

# required_grad is inherited
print(x.requires_grad)
print(y.requires_grad)
print(o.requires_grad)
