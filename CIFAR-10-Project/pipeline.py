from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate
import torchvision
from torchvision import transforms
from math import ceil, floor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Class and function definitions
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.layer1 = nn.Linear(16*5*5, 100)
        self.layer2 = nn.Linear(100, 50)
        self.layer3 = nn.Linear(50, 10)
    
    def forward(self, sample):
        sample = self.pool(functional.relu(self.conv1(sample)))
        sample = self.pool(functional.relu(self.conv2(sample)))
        sample = sample.flatten(start_dim=1, end_dim = 3)
        sample = functional.relu(self.layer1(sample))
        sample = functional.relu(self.layer2(sample))
        sample = self.layer3(sample)
        return sample




def train_model(n_epochs, learning_rate, train_dataloader, model, criterion, optimizer, classes, device):

    batch_size = train_dataloader.batch_size
    n_samples = len(train_dataloader.dataset)
    n_steps = ceil(n_samples/batch_size)

    print("\nStarting training...")

    open("./CIFAR-10-Project/validation_info.csv", "w").close() # clear file
    confusion_dict = {}
    confusion_dict["epoch"] = []
    confusion_dict["loss"] = []
    confusion_dict["matrices"] = []

    for epoch in range(n_epochs): 
        for step, (sample_batch, label_batch) in enumerate(train_dataloader):
            
            sample_batch, label_batch = sample_batch.to(device), label_batch.to(device)

            # forward pass and loss
            label_pred = model(sample_batch)
            loss = criterion(label_pred, label_batch)
            
            # empty gradient
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # propagate changes
            optimizer.step()
            # if ((epoch+1) % 1 == 0)&((step+1) % 256 == 0):
        print(f'epoch: {epoch+1}/{n_epochs}, loss = {loss.item():.4f}')

        with torch.no_grad():
            confusion_dict["epoch"].append(epoch+1)
            confusion_dict["loss"].append(loss.item())
            confusion = make_confusion_matrix(model, validation_dataloader, classes, device)
            confusion_dict["matrices"].append(confusion.tolist())
            confusion_df = pd.DataFrame(confusion_dict)
            confusion_df.to_csv("./CIFAR-10-Project/validation_info.csv")

    print("Finished training...")

@torch.no_grad()
def save_model(model, modelpath):
    print("\nSaving model state dict...")
    try:
        torch.save(model.state_dict(), modelpath)    # save model state dict
    except:
        print("Error saving model: check the save pathway is correct...")
    else:
        print("Model saved successfully...")

def load_model(ModelClass, modelpath):
    print("\nLoading model...")
    try:
        loaded_model = ModelClass()
        loaded_model.load_state_dict(torch.load(modelpath))
        loaded_model.eval() 
    except:
        print("Error loading model: check the load pathway is correct...")
    else:
        print("Model loaded successfully...")
        return loaded_model

@torch.no_grad()       
def make_confusion_matrix(model, dataloader, classes, device):
    confusion = torch.zeros(len(classes), len(classes), device=device)
    for sample_batch, label_batch in dataloader:
        sample_batch, label_batch = sample_batch.to(device), label_batch.to(device)
        sample_pred = model(sample_batch)

        _, batch_pred = torch.max(sample_pred, 1)
        for prediction, label in zip(batch_pred, label_batch):
            confusion[label,prediction] += 1
    return confusion

def plot_confusion_matrix(confusion, classes):
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion.cpu())
    fig.colorbar(cax)
    for (i, j), value in np.ndenumerate(confusion.cpu().numpy()):
        ax.text(j, i, f'{int(value)}', ha='center', va='center', color='white')
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Network guess')
    ax.set_ylabel('Correct label')
    ax.xaxis.set_ticks_position("bottom")
    plt.yticks(range(len(classes)), classes)
    plt.xticks(range(len(classes)), classes, rotation=90)


def get_class_accuracy(confusion):
    acc_list = []
    for label, row in enumerate(confusion):
        n_correct = row[label]
        n_samples = row.sum()
        acc_class = n_correct/n_samples
        rep = n_samples/torch.sum(confusion)
        acc_class, rep = acc_class.item(), rep.item()
        acc_list.append(acc_class)
    return acc_list

def get_total_accuracy(confusion):
    n_correct = torch.sum(torch.diag(confusion))
    n_samples = torch.sum(confusion)
    acc = n_correct/n_samples
    return acc 
# dataset and dataloader

## implementing dataset transform, data is initially a PILImage of range [0,1] we want to normalize the data
## to a range of [-1, 1] after transforming it to a tensor 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # scan for cuda availability

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

train_dataset = torchvision.datasets.CIFAR10(root="./CIFAR-10-Project/CIFAR-10-Dataset", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="./CIFAR-10-Project/CIFAR-10-Dataset", train=False, transform=transform)
test_dataset, validation_dataset = random_split(test_dataset, [ceil(len(test_dataset)/2), floor(len(test_dataset)/2)])

batch_size = 64
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, pin_memory=True)
validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, pin_memory=True)


classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
# model
model = CNNModel().to(device)

# loss and optimizer
learning_rate = 0.001
criterion = nn.CrossEntropyLoss() # cross entropy contains softmax
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training
n_epochs = 256

train_model(n_epochs, learning_rate, train_dataloader, model, criterion, optimizer, classes, device)

def checkpoint(model, checkpointpath):
    save_model(model, checkpointpath)

with torch.no_grad(): # Test accuracy
    modelpath = "./CIFAR-10-Project/model.pth"
    save_model(model, modelpath)
    # model = load_model(CNNModel, modelpath).to(device)

    confusion = make_confusion_matrix(model, test_dataloader, classes, device)
    plot_confusion_matrix(confusion, classes)
    plt.savefig("./CIFAR-10-Project/confusion_matrix.png", bbox_inches='tight')

    print("\nStarting accuracy test...")
    # print accuracy for each class
    acc_list = get_class_accuracy(confusion)
    for (acc, class_) in zip(acc_list, classes):
        print(f"{class_} accruacy = {100*acc:.2f}%")
    # print total accuracy
    acc = get_total_accuracy(confusion)
    print(f"\nTotal accruacy = {100*acc:.2f}%")
    plt.show()

