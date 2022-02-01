import torch
import torch.nn as nn
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from math import ceil

# dataset and dataloader

## implementing dataset transform, data is initially a PILImage of range [0,1] we want to normalize the data
## to a range of [-1, 1] after transforming it to a tensor 
transform = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root="./CIFAR-10-Project/CIFAR-10-Dataset", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="./CIFAR-10-Project/CIFAR-10-Dataset", train=False, transform=transform)

batch_size = 4
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
learning_rate = 0.01
n_samples = len(train_dataset)
criterion = nn.CrossEntropyLoss() # cross entropy contains softmax
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training
n_epochs = 4
n_steps = ceil(n_samples/batch_size)
print("Starting Training...")

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
        if ((epoch+1) % 1 == 0)&((step+1) % 2000 == 0):
            print(f'epoch: {epoch+1}/{n_epochs}, batch {step+1}/{n_steps},  loss = {loss.item():.4f}')

print("Finished Training...")

with torch.no_grad(): #Test accuracy
    for samples_test, labels_test in test_dataloader:

        n_correct = 0
        n_samples = 0

        acc = 100.0 * n_correct / n_samples
        print(f'accuracy = {acc}%')

    modelpath = "./CIFAR-10-Project/model.pth"
    torch.save(model.state_dict(), modelpath)    # save model state dict