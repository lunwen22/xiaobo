import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from vit_pytorch import ViT
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# 设置路径
data_folder = r"D:\lu\luone\juleijiaohdbscan\haochulihoushuju"
output_folder = r"D:\lu\luone\juleijiaohdbscan\VIT"  # 输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义数据集
class DepthDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        
        # 从 INI 文件中加载数据
        self.data = []
        for filename in os.listdir(data_folder):
            if filename.endswith('.ini'):
                with open(os.path.join(data_folder, filename), 'r') as file:
                    lines = file.readlines()
                    if len(lines) >= 2:
                        depth_value = float(lines[0].strip())
                        position_value = float(lines[1].strip())
                        self.data.append((filename, depth_value, position_value))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, depth_label, position_label = self.data[idx]
        
        # 生成随机一维信号并将其扩展为四维以适应 ViT 输入（例如，(batch_size, channels, height, width)）
        signal = torch.randn(1, 1, 8, 8)  # 生成随机信号，形状为 (1, 1, 8, 8)
        if self.transform:
            signal = self.transform(signal)
        
        return signal, torch.tensor(depth_label, dtype=torch.float32), torch.tensor(position_label, dtype=torch.float32)

# 定义模型
class ViTDepthModel(nn.Module):
    def __init__(self):
        super(ViTDepthModel, self).__init__()
        
        self.vit = ViT(
            image_size=8,  # 输入大小调整为适应一维信号的表示
            patch_size=2,   # Patch 大小
            num_classes=1,  # 输出一个连续值（深度值）
            dim=64,         # 嵌入维度
            depth=6,        # Transformer 层数
            heads=8,        # 注意力头的数量
            mlp_dim=128,    # MLP 的维度
            dropout=0.1,
            emb_dropout=0.1
        )
        
        # 输出层，分别预测深度值和深度点位位置
        self.depth_regressor = nn.Linear(64, 1)  # 输出深度值
        self.position_regressor = nn.Linear(64, 1)  # 输出深度点位位置

    def forward(self, x):
        x = self.vit(x)  # 使用 ViT
        depth_output = self.depth_regressor(x[:, 0, :])  # 使用分类 token 的输出进行深度值预测
        position_output = self.position_regressor(x[:, 0, :])  # 使用分类 token 的输出进行位置预测
        
        return depth_output, position_output

# 训练模型
def train_model(model, train_loader, criterion_depth, criterion_position, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for inputs, depth_labels, position_labels in train_loader:
            optimizer.zero_grad()
            depth_outputs, position_outputs = model(inputs)
            loss_depth = criterion_depth(depth_outputs, depth_labels.unsqueeze(1))
            loss_position = criterion_position(position_outputs, position_labels.unsqueeze(1))
            loss = loss_depth + loss_position  # 总损失为两个损失的和
            loss.backward()
            optimizer.step()

# 评估模型并保存结果
def evaluate_and_save_results(model, test_loader):
    model.eval()
    depth_predictions = []
    position_predictions = []
    true_depth_labels = []
    true_position_labels = []
    
    with torch.no_grad():
        for inputs, depth_labels, position_labels in test_loader:
            depth_outputs, position_outputs = model(inputs)
            depth_predictions.extend(depth_outputs.flatten().tolist())
            position_predictions.extend(position_outputs.flatten().tolist())
            true_depth_labels.extend(depth_labels.flatten().tolist())
            true_position_labels.extend(position_labels.flatten().tolist())
    
    # 保存预测值和真实值
    results_df = pd.DataFrame({
        'True Depth': true_depth_labels,
        'Predicted Depth': depth_predictions,
        'True Position': true_position_labels,
        'Predicted Position': position_predictions,
    })
    results_df.to_csv(os.path.join(output_folder, 'depth_position_predictions.csv'), index=False)

# 数据加载器
train_dataset = DepthDataset(data_folder)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 实例化模型和训练参数
model = ViTDepthModel()
criterion_depth = nn.MSELoss()  # 深度值的损失
criterion_position = nn.MSELoss()  # 点位位置的损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_model(model, train_loader, criterion_depth, criterion_position, optimizer, num_epochs=10)

# 评估并保存结果
evaluate_and_save_results(model, train_loader)