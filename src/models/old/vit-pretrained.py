import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader

# パラメータ
NUM_CLASSES = 2
BATCH_SIZE = 16
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データ前処理（サイズは224に合わせる必要あり）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# データセット読み込み
train_dataset = ImageFolder('/home/sota/research/sotaohnuma/bean/train', transform=transform)
val_dataset = ImageFolder('/home/sota/research/sotaohnuma/bean/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

weight_path = "/home/sota/research/sotaohnuma/coffee/vit/models/vit_b_16-c867db91.pth"

# モデルのインスタンスを作成
model = vit_b_16()
state_dict = torch.load(weight_path)
model.load_state_dict(state_dict)

# 出力層を差し替え（分類数に応じて）
in_features = model.heads[0].in_features
model.heads[0] = nn.Linear(in_features, NUM_CLASSES)

model = model.to(DEVICE)

# 学習設定
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# 学習ループ
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Accuracy: {acc:.4f}")

    # 検証
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total
    print(f" → Validation Accuracy: {val_acc:.4f}")

