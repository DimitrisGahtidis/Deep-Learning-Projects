import os, codecs
from re import I 
import numpy as np

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

modelpath = "./MNIST Project/model.pth"
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

class GrantSandersonModel(nn.Module):
    # this model is meant to reflect the 3 blue 1 brown model for a basic perceptron for MNIST digit recognition in the neural network series
    def __init__(self, input_size, output_size):
        super().__init__()

        # layers
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, output_size)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, sample):
        label_pred = self.sigmoid(self.layer1(sample))
        label_pred = self.sigmoid(self.layer2(label_pred))
        label_pred = self.sigmoid(self.layer3(label_pred))
        return label_pred


loaded_model = GrantSandersonModel(28*28, 10)
loaded_model.load_state_dict(torch.load(modelpath))
loaded_model.eval()

class TestDataset(Dataset):
    def __init__(self, data_dict):
        self.x = torch.from_numpy(data_dict['test_images'].astype(np.float32))
        self.y = torch.from_numpy(data_dict['test_labels'].astype(np.float32))
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

test_dataset = TestDataset(data_dict)

batch_size = 1000
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

def plot_image(sample):
    plt.imshow(sample, cmap='gray')

for samples_test, labels_test in test_dataloader:
    prediction = loaded_model(torch.flatten(samples_test[0]))
    prediction = torch.argmax(prediction)
    label = int(labels_test[0])
    plot_image(samples_test[0])
    plt.title(f'Value: {label}, NN Guess: {prediction}')
    plt.show()