import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据文件
ini_file_path = r'D:\shudeng\波形图\log\126331124553553_OriData_20230728151315.ini'
with open(ini_file_path, 'r') as file:
    data = np.array([float(line.strip()) for line in file if line.strip()])

# 定义带通滤波器
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# 增强的小波去噪函数
def enhanced_wavelet_denoising(data, wavelet='coif5', level=6):  # 将 level 设置为 6，使用 coif5
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # 设置阈值，去除高频噪声
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
    
    return pywt.waverec(coeffs, wavelet)

# 假设采样率
fs = 1000  # 例如采样率为1000Hz

# 固定 lowcut 为 8，遍历 highcut 从 8 到 10，步长为 1
lowcut = 8
for highcut in range(8, 11, 1):  # 遍历 highcut 8到10，进一步减小
    if lowcut >= highcut:  # 保证 lowcut 小于 highcut
        continue

    # 使用带通滤波器对信号进行初步处理
    filtered_signal = bandpass_filter(data, lowcut, highcut, fs)

    # 使用小波增强去噪进一步处理信号
    enhanced_denoised_signal = enhanced_wavelet_denoising(filtered_signal)

    # 绘制去噪前后的对比图
    plt.figure(figsize=(12, 6))

    # 原始信号
    plt.subplot(2, 1, 1)
    plt.plot(data)
    plt.title(f'原始动液面信号 - lowcut {lowcut}, highcut {highcut}')

    # 处理后的信号
    plt.subplot(2, 1, 2)
    plt.plot(enhanced_denoised_signal)
    plt.title(f'带通滤波与小波增强去噪后的动液面信号 - lowcut {lowcut}, highcut {highcut}')
    plt.tight_layout()

    # 直接显示图像
    plt.show()
