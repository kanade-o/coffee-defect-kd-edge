import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize((180, 180)),
	torchvision.transforms.ToTensor(),
	torchvision.transforms.Normalize((0.5,), (0.5,))
])

train_dataset = ImageFolder('../bean/train', transform=trans)
val_dataset = ImageFolder('../bean/val', transform=trans)
test_dataset = ImageFolder('../bean/test', transform=trans)

epochs = 100
num_classes = 2
batch_size = 10

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*45*45, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda:0")
net = CNN()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

train_loss_value=[]
train_acc_value=[]
test_loss_value=[]
test_acc_value=[]

for epoch in range(epochs):
    print('epoch', epoch+1)
    sum_loss = 0.0
    sum_correct = 0
    sum_total = 0

    for (inputs, labels) in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()
        _, predicted = outputs.max(1)
        sum_total += labels.size(0)
        sum_correct += (predicted == labels).sum().item()
        loss.backward()
        optimizer.step()
    print("train mean loss={}, accuracy={}".format(sum_loss*batch_size/len(train_loader.dataset), float(sum_correct/sum_total)))
    train_loss_value.append(sum_loss*batch_size/len(train_loader.dataset))
    train_acc_value.append(float(sum_correct/sum_total))

    sum_loss = 0.0
    sum_correct = 0
    sum_total = 0

    for (inputs, labels) in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()
        _, predicted = outputs.max(1)
        sum_total += labels.size(0)
        sum_correct += (predicted == labels).sum().item()
    print("test mean loss={}, accuracy={}".format(sum_loss*batch_size/len(val_loader.dataset), float(sum_correct/sum_total)))
    test_loss_value.append(sum_loss*batch_size/len(val_loader.dataset))
    test_acc_value.append(float(sum_correct/sum_total))
 
print(train_loss_value)
print("ACCCCCCCCCCCCCCCCCCCCCCC")
print(train_acc_value)

plt.figure(figsize=(6,6))

plt.plot(range(epochs), train_loss_value)
plt.plot(range(epochs), test_loss_value, c='#00ff00')
plt.xlim(0, epoch)
plt.ylim(0, 2.5)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train loss', 'test loss'])
plt.title('loss')
plt.savefig("loss_image.png")
plt.clf()

plt.plot(range(epochs), train_acc_value)
plt.plot(range(epochs), test_acc_value, c='#00ff00')
plt.xlim(0, epoch)
plt.ylim(0, 2.5)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train acc', 'test acc'])
plt.title('accuracy')
plt.savefig("acc_image.png")
plt.clf()

