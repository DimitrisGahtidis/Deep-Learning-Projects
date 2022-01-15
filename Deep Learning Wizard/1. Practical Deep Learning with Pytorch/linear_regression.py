import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# matplotlib inline

# Creates 50 random x and y numbers
np.random.seed(1)
n = 50
x = np.random.randn(n)
y = x * np.random.randn(n)

# Makes the dots colorful
colors = np.random.rand(n)

# Plots best-fit line via polyfit
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

# Plots the random x and y data points we created
# Interestingly, alpha makes it more aesthetically pleasing
plt.scatter(x, y, c=colors, alpha=0.5)
plt.show()

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

# Basic model structure:
# 1. Linear model
#   True equation: y = 2x+1
# 2. Forward
#   Example
#       input: x=1
#       output y=?

# Create class


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


# instantiate a model class
input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)

# instantiate a loss class
# MSE loss = mean squared error
# MSE = (1/n) sum_1^n (yhat_i - y_i)^2
# y_hat is the prediction
# y is the true values that is produced

criterion = nn.MSELoss()

# instantiate an optimizer class
# theta = theta - eta \dot grad_theta
# theta represents the parameters (our variables)
# eta is a learning rate proportionality (how fast you want to learn)
# grad_theta is the gradient of the parameters

# Even simpler we have
# parameters = parameters - learning_rate * parameters_gradients
# where in our case parameters is alpha and beta in y = alpha*x+beta
# our desired parameters are alpha = 2 and beta = 1 in y=2x+1
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# train the model
# 1 epoch: goes through x_train once
# 100 epochs:
#   100x mapping x_train = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Process
# a. convert the inputs/labels to tensors with gradients
# b. clear gradients buffers
# c. Get output given inputs
# d. get the loss of the output
# e. get gradients w.r.t. parameters
# f. update parameters using gradients:
#   this is done using:
#   parameters = parameters - learning_rate * paramters_gradients
# g. REPEAT

epochs = 100
for epoch in range(epochs):
    epoch += 1
    # convert numpy array to torch Variable
    inputs = torch.from_numpy(x_train).requires_grad_()
    labels = torch.from_numpy(y_train)

    # clear gradients w.r.t. parameters
    optimizer.zero_grad()

    # forward to get output
    outputs = model(inputs)

    # calculate loss
    loss = criterion(outputs, labels)

    # getting gradients w.r.t. parameters
    loss.backward()

    # updating parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))


# looking at the predicted values
predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
print(type(predicted))

# we can compare these predicted values to the true ones
print(y_train)

# and plot the predicted vs the actual values

# Clear figure
plt.clf()

# Get predictions
predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()

# Plot true data
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)

# Plot predictions
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)

# Legend and plot
plt.legend(loc='best')
plt.show()

# saving the model
save_model = False
if save_model is True:
    # save only parameters
    # alpha & beta
    torch.save(model.state_dict(), 'awesome_model.pk1')

# loading a model
load_model = False
if load_model is True:
    model.load_state_dict(torch.load('awesome_model.pk1'))
