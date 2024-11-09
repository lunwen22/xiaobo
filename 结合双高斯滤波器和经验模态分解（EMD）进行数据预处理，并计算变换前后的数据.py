import numpy as np
from scipy.ndimage import gaussian_filter1d
from PyEMD import EMD
import matplotlib.pyplot as plt

def double_gaussian_filter(signal, sigma1, sigma2):
    smooth_signal_1 = gaussian_filter1d(signal, sigma1)
    smooth_signal_2 = gaussian_filter1d(signal, sigma2)
    return (smooth_signal_1 + smooth_signal_2) / 2

def apply_emd(signal):
    emd = EMD()
    imfs = emd.emd(signal)
    return imfs

# 示例数据
data = np.array([-1581.3766871861400, -1755.3721780362200, -1760.8676715268600, 
                 -1704.3631677511600, -1755.8586668021700, -1890.8541687728600, 
                 -1751.8496737561600, -1677.3451818449100,-1815.3406931319300])

# 应用双高斯滤波
sigma1, sigma2 = 1, 3
smoothed_data = double_gaussian_filter(data, sigma1, sigma2)

# 应用EMD
imfs = apply_emd(smoothed_data)

# 计算重构信号
reconstructed_signal = np.sum(imfs, axis=0)

# 打印结果
print("原始数据：", data)
print("平滑后的数据：", smoothed_data)
print("重构后的数据：", reconstructed_signal)

# 可视化
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(data, label='Original Data')
plt.title('Original Data')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(smoothed_data, label='Smoothed Data (Double Gaussian)')
plt.title('Smoothed Data (Double Gaussian)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(reconstructed_signal, label='Reconstructed Signal (EMD)')
plt.title('Reconstructed Signal (EMD)')
plt.legend()

plt.tight_layout()
plt.show()

# 可视化IMFs
plt.figure(figsize=(12, 8))
for i, imf in enumerate(imfs):
    plt.subplot(len(imfs), 1, i+1)
    plt.plot(imf, label=f'IMF {i+1}')
    plt.legend()
plt.tight_layout()
plt.show()
