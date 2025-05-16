import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from vit_pytorch import ViT
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

NUM_EPOCHS = 100
BATCH_SIZE = 10
NUM_CLASSES = 2

trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize((180, 180)),
	torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = ImageFolder('/home/sota/research/sotaohnuma/bean/train', transform=trans)
val_dataset = ImageFolder('/home/sota/research/sotaohnuma//bean/val', transform=trans)
test_dataset = ImageFolder('/home/sota/research/sotaohnuma/bean/test', transform=trans)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViT(
    image_size=180,
    patch_size=15,
    num_classes=2,
    dim=256,
    depth=4,
    heads=6,
    mlp_dim=256,
    dropout=0.2,
    emb_dropout=0.1
).to(device)

criterion = torch.nn.CrossEntropyLoss()  # 例
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

try:
    for epoch in range(NUM_EPOCHS):
        print(f"epoch: {epoch}")
        epoch_train_loss = 0
        epoch_train_acc = 0
        epoch_test_loss = 0
        epoch_test_acc = 0

        model.train()
        for data in train_loader:
            try:
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_train_loss += loss.item() / len(train_loader)
                acc = (outputs.argmax(dim=1) == labels).float().mean().cpu()
                epoch_train_acc += acc / len(train_loader)

                loss.backward()
                optimizer.step()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("[Warning] CUDA OOM. Skipping this batch.")
                    torch.cuda.empty_cache()
                else:
                    raise e

        model.eval()
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_test_loss += loss.item() / len(val_loader)
                test_acc = (outputs.argmax(dim=1) == labels).float().mean().cpu()
                epoch_test_acc += test_acc / len(val_loader)

        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_acc)

        print(f'Epoch {epoch+1}: train acc. {epoch_train_acc:.2f} train loss {epoch_train_loss:.2f}')
        print(f'Epoch {epoch+1}: valid acc. {epoch_test_acc:.2f} valid loss {epoch_test_loss:.2f}')

except Exception as e:
    print("Training crashed:", str(e))
    torch.save(model.state_dict(), 'crashed_model.pth')

# グラフ保存（保存失敗対策）
try:
    os.makedirs("plots", exist_ok=True)

    plt.figure()
    plt.plot(range(NUM_EPOCHS), train_losses, label='Train')
    plt.plot(range(NUM_EPOCHS), test_losses, label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Valid Loss')
    plt.legend()
    plt.grid()
    plt.savefig("plots/loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(range(NUM_EPOCHS), train_accuracies, label='Train')
    plt.plot(range(NUM_EPOCHS), test_accuracies, label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Valid Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig("plots/accuracy_curve.png")
    plt.close()
except Exception as e:
    print("Failed to save plots:", str(e))
