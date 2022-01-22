import torch

# autograd is a package that does gradient computations

x = torch.randn(3, requires_grad=True)
# this tells pytorch that it will need to calculate the gradient for a function with this variable as an input later
print(x)
y = x+2
print(y) # y has the grad_fn attribute
z = y*y*2
print(z) # same with z
# v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
# z.backward(v) performs the vector jacobian product J dot v
z = z.mean()
print(z)
z.backward() # calculates dz/dx, backward has no argument because it has a scalar output
print(x.grad)

x = torch.randn(3, requires_grad=True)
print(x)

# you can create a variable from x that does not require grad as follows
# x.requires_grad_(False)
# x.detach()
# with torch.no_grad(): 