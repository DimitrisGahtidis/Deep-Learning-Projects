import torch 
import numpy as np 

# prior tensor basics are covered in the deep learning wizard folder

# how to create and move tensors in the cpu using cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device) # create tensor in device directly
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    # z.numpy() throws an error because numpy can only handle cpu tensors
    z = z.to("cpu")


x = torch.ones(5, requires_grad=True) 
# this tells pytorch that it will need to calculate the gradient for a function with this variable as an input later
print(x)