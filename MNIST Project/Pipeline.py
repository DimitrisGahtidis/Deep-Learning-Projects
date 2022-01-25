import os, codecs
from re import I 
import numpy as np

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader


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

# 0) Create dataset and dataloader
class TrainDataset(Dataset):
    def __init__(self, data_dict):
        self.x = torch.from_numpy(data_dict['train_images'].astype(np.float32))
        self.y = torch.from_numpy(data_dict['train_labels'].astype(np.float32))
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

class TestDataset(Dataset):
    def __init__(self, data_dict):
        self.x = torch.from_numpy(data_dict['test_images'].astype(np.float32))
        self.y = torch.from_numpy(data_dict['test_labels'].astype(np.float32))
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

train_dataset = TrainDataset(data_dict)
test_dataset = TestDataset(data_dict)

batch_size = 100
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.n_samples)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # scan for cuda availability

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

n_samples = train_dataset.n_samples
model = GrantSandersonModel(28*28, 10).to(device)

# 2) loss and optimizer
learning_rate = 0.1
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training
n_epochs = 100

for epoch in range(n_epochs):
    for i, (x_batch, y_batch) in enumerate(train_dataloader):
        
        # flatten 28x28 image array to 784 column tensor and move to cuda device
        x_batch = torch.flatten(x_batch, start_dim=1).to(device)
        y_batch = y_batch.to(device)

        # e.g. if label = 2 replace it with a tensor of the form [0, 0, 1, 0, ...] (1x10) 
        empty = torch.zeros(batch_size, 10) # create empty zeros array tensor for each sample
        activation_index = [[*range(0,batch_size)],[int(y) for y in y_batch]] # index to mark where the 1's should be placed
        empty[activation_index] = 1 
        y_batch = empty

        # forward pass and loss
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        
        # empty gradient
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # propagate changes
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}/{n_epochs},  loss = {loss.item():.4f}')

with torch.no_grad(): #Test accuracy
    for i, (x_test, y_test) in enumerate(test_dataloader):

        x_test = torch.flatten(x_test, start_dim=1).to(device)
        y_test = y_test.to(device)

        n_correct = 0
        n_samples = y_test.shape[0]

        y_pred = model(x_test)

        _, predictions = torch.max(y_pred, 1)
        labels = y_test

        n_correct += (predictions == labels).sum().item()
        acc = 100.0 * n_correct / n_samples
        print(f'accuracy = {acc}%')