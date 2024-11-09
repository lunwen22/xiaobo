import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 双高斯滤波器
def double_gaussian_filter(signal, sigma1, sigma2):
    smooth_signal_1 = gaussian_filter1d(signal, sigma1)
    smooth_signal_2 = gaussian_filter1d(signal, sigma2)
    return (smooth_signal_1 + smooth_signal_2) / 2

# 液体状态层
class LiquidStateLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiquidStateLayer, self).__init__()
        self.hidden_size = hidden_size
        self.recurrent = nn.Linear(hidden_size, hidden_size)
        self.input = nn.Linear(input_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_size)
        for t in range(x.size(1)):
            h = self.activation(self.recurrent(h) + self.input(x[:, t].unsqueeze(1)))
        return h

# 液面神经网络
class LiquidStateNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LiquidStateNN, self).__init__()
        self.liquid_state_layer = LiquidStateLayer(input_size, hidden_size)
        self.readout_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.liquid_state_layer(x)
        output = self.readout_layer(h)
        return output

# 读取油井动液面数据
def load_data(file_path):
    data = np.loadtxt(file_path)
    return data

# 数据预处理
def preprocess_data(data):
    # 数据归一化
    data = (data - data.mean()) / data.std()
    return data

# 示例数据处理和可视化
def main():
    # 读取数据
    data_path = 'F:\\ProofingTool\\数据\\141361363626636_OriData_20220801154803.ini'
    data = load_data(data_path)

    # 数据预处理
    data = preprocess_data(data)

    # 分割数据
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    # 打印数据形状
    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)

    # 转换为张量
    train_tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(0)
    test_tensor = torch.tensor(test_data, dtype=torch.float32).unsqueeze(0)

    # 打印张量形状
    print("Train tensor shape:", train_tensor.shape)
    print("Test tensor shape:", test_tensor.shape)

    # 模型参数
    input_size = 1  # 每次输入一个时间点的值
    hidden_size = 50
    output_size = 1

    # 创建模型
    model = LiquidStateNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_tensor)
        loss = criterion(output, train_tensor[:, -1, :output_size])
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 测试模型
    model.eval()
    with torch.no_grad():
        test_output = model(test_tensor)
        print(test_output)

    # 可视化结果
    plt.plot(test_output.numpy().flatten(), label='Predicted')
    plt.plot(test_tensor.numpy().flatten(), label='True')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
