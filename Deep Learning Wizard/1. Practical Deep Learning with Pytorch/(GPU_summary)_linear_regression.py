import torch
import torch.nn as nn
import numpy as np
# Just remember always 2 things must be on GPU
# model
# tensors

# Build a toy dataset

x_values = [i for i in range(11)]
print(x_values)

# convert list to numpy array
x_train = np.array(x_values, dtype=np.float32)
print(x_train.shape)

# convert to 2-dimensional array
# IMPORTANT: 2D required
x_train = x_train.reshape(-1, 1)
print(x_train.shape)

y_values = [2*i + 1 for i in x_values]
print(y_values)

# converting to numpy again
y_train = np.array(y_values, dtype=np.float32)
print(y_train.shape)

# IMPORTANT: 2D required
y_train = y_train.reshape(-1, 1)
print(y_train.shape)

'''
STEP 1: CREATE MODEL CLASS
'''


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


'''
STEP 2: INSTANTIATE MODEL CLASS
'''
input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)


#######################
#  USE GPU FOR MODEL  #
#######################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

'''
STEP 3: INSTANTIATE LOSS CLASS
'''

criterion = nn.MSELoss()

'''
STEP 4: INSTANTIATE OPTIMIZER CLASS
'''

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
STEP 5: TRAIN THE MODEL
'''
epochs = 100
for epoch in range(epochs):
    epoch += 1
    # Convert numpy array to torch Variable

    #######################
    #  USE GPU FOR MODEL  #
    #######################
    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_train).to(device)

    # Clear gradients w.r.t. parameters
    optimizer.zero_grad()

    # Forward to get output
    outputs = model(inputs)

    # Calculate Loss
    loss = criterion(outputs, labels)

    # Getting gradients w.r.t. parameters
    loss.backward()

    # Updating parameters
    optimizer.step()

    # Logging
    print('epoch {}, loss {}'.format(epoch, loss.item()))
