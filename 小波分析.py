import numpy as np
import pywt
import matplotlib.pyplot as plt

# 读取数据
file_path = r'D:\shudeng\ProofingTool\数据\215711126520129_OriData_20221102055610(1355).ini'
data = np.loadtxt(file_path)
time = np.arange(len(data))
signal = data

# 定义小波去噪函数
def wavelet_denoising(signal, wavelet_name, level=6):
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    threshold = np.sqrt(2 * np.log(len(signal))) * np.median(np.abs(coeffs[-1])) / 0.6745
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet_name)

# 使用 db4 和 coif5 进行去噪
denoised_db4 = wavelet_denoising(signal, 'db4')
denoised_coif5 = wavelet_denoising(signal, 'coif5')

# 确保去噪信号和原始信号长度一致
min_length = min(len(time), len(denoised_db4), len(denoised_coif5))
time = time[:min_length]
signal = signal[:min_length]
denoised_db4 = denoised_db4[:min_length]
denoised_coif5 = denoised_coif5[:min_length]

# 可视化
plt.figure(figsize=(12, 8))

# 原始信号子图
plt.subplot(3, 1, 1)
plt.plot(time, signal, label='Original Signal', color='blue')
plt.title('Original Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# db4 小波去噪后的信号子图
plt.subplot(3, 1, 2)
plt.plot(time, denoised_db4, label='Denoised with db4', color='orange')
plt.title('Denoised Signal with db4')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# coif5 小波去噪后的信号子图
plt.subplot(3, 1, 3)
plt.plot(time, denoised_coif5, label='Denoised with coif5', color='green')
plt.title('Denoised Signal with coif5')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# 调整布局并显示
plt.tight_layout()
plt.show()
