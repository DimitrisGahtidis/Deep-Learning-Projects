import os, codecs
from re import I 
import numpy as np
import torch
import torch.nn as nn 


# --------------------------Gosh4AI-Data-Processing-Code--------------------------------
# PROVIDE YOUR DIRECTORY WITH THE EXTRACTED FILES HERE
datapath = './MNIST Project/MNIST Dataset/'

files = os.listdir(datapath)

def get_int(b):   # CONVERTS 4 BYTES TO A INT
    return int(codecs.encode(b, 'hex'), 16)

data_dict = {}
for file in files:
    if file.endswith('ubyte'):  # FOR ALL 'ubyte' FILES
        print('Reading ',file)
        with open (datapath+file,'rb') as f:
            data = f.read()
            type = get_int(data[:4])   # 0-3: THE MAGIC NUMBER TO WHETHER IMAGE OR LABEL
            length = get_int(data[4:8])  # 4-7: LENGTH OF THE ARRAY  (DIMENSION 0)
            if (type == 2051):
                category = 'images'
                num_rows = get_int(data[8:12])  # NUMBER OF ROWS  (DIMENSION 1)
                num_cols = get_int(data[12:16])  # NUMBER OF COLUMNS  (DIMENSION 2)
                parsed = np.frombuffer(data,dtype = np.uint8, offset = 16)  # READ THE PIXEL VALUES AS INTEGERS
                parsed = parsed.reshape(length,num_rows,num_cols)  # RESHAPE THE ARRAY AS [NO_OF_SAMPLES x HEIGHT x WIDTH]           
            elif(type == 2049):
                category = 'labels'
                parsed = np.frombuffer(data, dtype=np.uint8, offset=8) # READ THE LABEL VALUES AS INTEGERS
                parsed = parsed.reshape(length)  # RESHAPE THE ARRAY AS [NO_OF_SAMPLES]                           
            if (length==10000):
                set = 'test'
            elif (length==60000):
                set = 'train'
            data_dict[set+'_'+category] = parsed  # SAVE THE NUMPY ARRAY TO A CORRESPONDING KEY  
# --------------------------------------------------------------------------------------

# 0) prepare data
def flatten_image(x):
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = torch.flatten(x, start_dim=1)
    return 
# load data
x_train = data_dict['train_images'].astype(np.float32)
x_test = data_dict['test_images'].astype(np.float32)
y_train = data_dict['train_labels'].astype(np.float32)
y_test = data_dict['test_labels'].astype(np.float32)

# convert data to tensor
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

# flatten 28x28 image array to 784 column tensor
x_train = torch.flatten(x_train, start_dim=1)
x_test = torch.flatten(x_test, start_dim=1)

def create_labels(y):
    n_labels = y.shape[0]
    empty = torch.zeros(n_labels, 10) # create a n_labelsx10 empty label matrix
    for i, digit in enumerate(y):
        empty[i][int(digit.item())] = 1 # attach label (1 flag) to appropriate index e.g. a "5" looks like [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    return empty

y_train = create_labels(y_train)
y_test = create_labels(y_test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # scan for cuda availability

x_train = x_train.to(device)
x_test = x_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)


# 1) model
class GrantSandersonModel(nn.Module):
    # this model is meant to reflect the 3 blue 1 brown model for a basic perceptron for MNIST digit recognition in the neural network series
    def __init__(self, input_size, output_size):
        super().__init__()

        # layers
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, output_size)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        y_pred = self.sigmoid(self.layer1(x))
        y_pred = self.sigmoid(self.layer2(y_pred))
        y_pred = self.sigmoid(self.layer3(y_pred))
        return y_pred

n_samples, n_features = x_train.shape
model = GrantSandersonModel(n_features, 10).to(device)

# 2) loss and optimizer
learning_rate = 0.05
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training
n_epochs = 400
batch_size = 100
iterations = int(n_samples/batch_size)

x_batch = torch.split(x_train, batch_size)
y_batch = torch.split(y_train, batch_size)

for epoch in range(n_epochs):
    for i in range(iterations):
        # forward pass and loss
        y_pred = model(x_batch[i])
        loss = criterion(y_pred, y_batch[i])
        
        # empty gradient
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # propagate changes
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}/{n_epochs},  loss = {loss.item():.4f}')

