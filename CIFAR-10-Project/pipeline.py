import torch
import torch.nn as nn
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from math import ceil
import matplotlib.pyplot as plt
import numpy as np

# dataset and dataloader

## implementing dataset transform, data is initially a PILImage of range [0,1] we want to normalize the data
## to a range of [-1, 1] after transforming it to a tensor 
transform = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root="./CIFAR-10-Project/CIFAR-10-Dataset", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="./CIFAR-10-Project/CIFAR-10-Dataset", train=False, transform=transform)

batch_size = 10
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # scan for cuda availability

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
# model
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

model = CNNModel().to(device)

# loss and optimizer
learning_rate = 0.005
n_samples = len(train_dataset)
criterion = nn.CrossEntropyLoss() # cross entropy contains softmax
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training
n_epochs = 15
n_steps = ceil(n_samples/batch_size)


def train_model(n_epochs, learning_rate, train_dataloader, model, criterion, optimizer, device):

    batch_size = train_dataloader.batch_size
    n_samples = len(train_dataloader.dataset)
    n_steps = ceil(n_samples/batch_size)

    print("\nStarting training...")
    for epoch in range(n_epochs):
        for step, (sample_batch, label_batch) in enumerate(train_dataloader):
            
            # flatten 28x28 image array to 784 column tensor and move to cuda device
            sample_batch = sample_batch.to(device)
            label_batch = label_batch.to(device)

            # forward pass and loss
            label_pred = model(sample_batch)
            loss = criterion(label_pred, label_batch)
            
            # empty gradient
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # propagate changes
            optimizer.step()
            if ((epoch+1) % 1 == 0)&((step+1) % 1000 == 0):
                print(f'epoch: {epoch+1}/{n_epochs} batch = {step+1}/{n_steps} loss = {loss.item():.4f}')
    print("Finished training...")

train_model(n_epochs, learning_rate, train_dataloader, model, criterion, optimizer, device)

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
        sample_batch = sample_batch.to(device)
        label_batch = label_batch.to(device)
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

with torch.no_grad(): # Test accuracy
    modelpath = "./CIFAR-10-Project/model.pth"
    save_model(model, modelpath)
    # model = load_model(CNNModel, modelpath).to(device)
    confusion = make_confusion_matrix(model, test_dataloader, classes, device)
    
    plot_confusion_matrix(confusion, classes)
    plt.savefig("./CIFAR-10-Project/confusion_matrix.png", bbox_inches='tight')

    print("\nStarting accuracy test...")
    n_correct = 0
    n_samples = 0

    n_correct_class = [0 for i in range(10)]
    n_samples_class = [0 for i in range(10)]

    for sample_batch, label_batch in test_dataloader:
        sample_batch = sample_batch.to(device)
        label_batch = label_batch.to(device)
        sample_pred = model(sample_batch)

        _, batch_pred = torch.max(sample_pred, 1)
        n_samples = label_batch.shape[0]
        n_correct = (batch_pred == label_batch).sum().item()

        for prediction, label in zip(batch_pred, label_batch):
            if (label == prediction):
                n_correct_class[label] += 1
            n_samples_class[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Total network accuracy = {acc}%')

    for n_correct, n_samples, class_ in zip(n_correct_class, n_samples_class, classes):
        acc_class = 100.0 * n_correct/n_samples
        rep = 100.0 * n_samples/sum(n_samples_class)
        print(f'{class_} accuracy = {acc_class}% representation = {100*n_samples/sum(n_samples_class)}%')
    plt.show()

