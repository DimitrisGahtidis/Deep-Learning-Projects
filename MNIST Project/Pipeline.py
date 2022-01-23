import os, codecs 
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
x_test = torch.flatten(x_train, start_dim=1)


# 1) model

class GrantSandersonModel(nn.Module):
    # this model is meant to reflect the 3 blue 1 brown model for a basic perceptron for MNIST digit recognition in the neural network series
    def __init__(self, input_size, output_size):
        super().__init__()

        # layers
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16,16)
        self.layer3 = nn.Linear(16,output_size)
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.layer1(x))
        y_pred = torch.sigmoid(self.layer2(x))
        y_pred = torch.sigmoid(self.layer3(x))
        return y_pred

n_samples, n_features = x_train.shape
model = GrantSandersonModel(n_features, 10)

# 2) loss and optimizer
# 4) training