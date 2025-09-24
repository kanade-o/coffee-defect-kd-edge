import torch
import torch.nn as nn

class cnn_student100(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super(cnn_student100, self).__init__()
        
        # --- ここでモデルの容量（サイズ）を調整します ---
        
        # ブロック1: 224x224 -> 112x112 -> 56x56
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 112x112
        )
        
        # ブロック2: 56x56 -> 28x28
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 28x28
        )
        
        # ブロック3: 28x28 -> 14x14
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 14x14
        )
        
        # 分類器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 特徴マップのサイズを1x1に変換
            nn.Flatten(),
            nn.Linear(in_features=32, out_features=num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

