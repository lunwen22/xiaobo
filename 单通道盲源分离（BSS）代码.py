import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# 读取数据
file_path = r'D:\shudeng\ProofingTool\数据\46-03-6.ini'
data = np.loadtxt(file_path)  # 假设多通道数据
time = np.arange(len(data))

# 模拟多通道信号，假设数据只有一个通道，为了ICA增加一个通道
multi_signal = np.column_stack([data, data * 0.5 + np.random.normal(0, 0.1, len(data))])

# FastICA分离
ica = FastICA(n_components=2)  # 分离成两个独立分量
S_ = ica.fit_transform(multi_signal)  # 估计源信号

# 可视化独立分量
plt.figure(figsize=(12, 8))

# 独立成分1
plt.subplot(2, 1, 1)
plt.plot(time, S_[:, 0], label='Independent Component 1', color='b', linewidth=1.5)
plt.title('Independent Component 1', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True)  # 添加网格线
plt.xlim([0, len(time)])  # 设置x轴范围
plt.ylim([min(S_[:, 0]) * 1.1, max(S_[:, 0]) * 1.1])  # 设置y轴范围，给出一定的间隔

# 独立成分2
plt.subplot(2, 1, 2)
plt.plot(time, S_[:, 1], label='Independent Component 2', color='g', linestyle='--', linewidth=1.5)
plt.title('Independent Component 2', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True)  # 添加网格线
plt.xlim([0, len(time)])  # 设置x轴范围
plt.ylim([min(S_[:, 1]) * 1.1, max(S_[:, 1]) * 1.1])  # 设置y轴范围

# 调整布局并显示
plt.tight_layout()

# 保存图像
plt.savefig('ica_components.png', dpi=300)  # 保存为高分辨率的图片
plt.show()
