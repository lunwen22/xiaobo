import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 假设你已经读取了数据，并存储在变量 data 中
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
def enhanced_wavelet_denoising(data, wavelet='coif5', level=1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # 设置阈值，去除高频噪声
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
    
    return pywt.waverec(coeffs, wavelet)

# 假设采样率
fs = 1000  # 例如采样率为1000Hz

# 调整滤波器的参数，根据信号的特点，去掉红圈噪声
lowcut = 10.0   # 低频截止点，去除过低频噪声
highcut = 100.0  # 高频截止点，去掉红圈噪声，保留绿圈信号

# 使用带通滤波器对信号进行初步处理
filtered_signal = bandpass_filter(data, lowcut, highcut, fs)

# 使用小波增强去噪进一步处理信号
enhanced_denoised_signal = enhanced_wavelet_denoising(filtered_signal)

# 绘制去噪前后的对比图
plt.figure(figsize=(12, 6))

# 原始信号
plt.subplot(2, 1, 1)
plt.plot(data)
plt.title('原始动液面信号')

# 处理后的信号，减少红圈噪声，保留绿圈信号
plt.subplot(2, 1, 2)
plt.plot(enhanced_denoised_signal)
plt.title('带通滤波与小波增强去噪后的动液面信号')
plt.tight_layout()
plt.show()
