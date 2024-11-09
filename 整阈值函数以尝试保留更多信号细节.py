import numpy as np
import pywt
import matplotlib.pyplot as plt

def soft_threshold(w, lambd, scale=1.2):
    """应用软阈值，增加一个比例因子以增强去噪效果"""
    return np.sign(w) * np.maximum(np.abs(w) - lambd * scale, 0)

def adaptive_threshold(w, sigma, alpha=2.0):
    """使用提供的alpha增强去噪"""
    N = len(w)
    return sigma * np.sqrt(2 * np.log(N) * alpha)

def enhanced_soft_threshold(w, lambd, proportion=0.6):
    """增加动态阈值调整以保留更多信号细节"""
    adjusted_lambd = lambd * (1 - proportion + proportion * np.exp(-np.abs(w) / lambd))
    return np.sign(w) * np.maximum(np.abs(w) - adjusted_lambd, 0)

def wavelet_denoising(signal, wavelet='coif5', level=None, alpha=2.0):
    """小波去噪处理，增加了动态阈值处理"""
    if level is None:
        level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet))
    coeffs = pywt.wavedec(signal, wavelet, level=min(level, 5))
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thresholded_coeffs = [coeffs[0]]
    for coeff in coeffs[1:]:
        lambd = adaptive_threshold(coeff, sigma, alpha)
        thresholded_coeffs.append(enhanced_soft_threshold(coeff, lambd))
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet)
    return denoised_signal[:len(signal)]

def read_signal_data(file_path):
    """读取信号数据，处理可能的文件读取错误"""
    try:
        with open(file_path, 'r') as file:
            signal = np.array([float(line.strip()) for line in file if line.strip()])
        return signal
    except Exception as e:
        print(f"Error reading data: {e}")
        return None

# 指定文件路径
file_path = r"D:\shudeng\ProofingTool\数据\215711126520129_OriData_20221102055610(1355).ini"

# 从文件读取数据
noisy_signal = read_signal_data(file_path)

if noisy_signal is not None:
    # 小波去噪
    denoised_signal = wavelet_denoising(noisy_signal, alpha=2.0)

    # 绘制去噪前后的信号对比
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(noisy_signal, label='Original (Noisy) Signal')
    plt.title('Original (Noisy) Signal')
    plt.xlabel('Sample index')
    plt.ylabel('Signal amplitude')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(denoised_signal, label='Denoised Signal')
    plt.title('Denoised Signal')
    plt.xlabel('Sample index')
    plt.ylabel('Signal amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()
