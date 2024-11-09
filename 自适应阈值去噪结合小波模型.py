import numpy as np
import pywt
import matplotlib.pyplot as plt

def soft_threshold(w, lambd, scale=1.2):
    """增加一个比例因子以增强去噪效果"""
    return np.sign(w) * np.maximum(np.abs(w) - lambd * scale, 0)

def adaptive_threshold(w, sigma, alpha):
    """使用提供的alpha增强去噪"""
    N = len(w)
    return sigma * np.sqrt(2 * np.log(N) * alpha)

def wavelet_denoising(signal, wavelet='db8', level=None, alpha=2.0):
    """小波去噪处理，自动确定层数"""
    if level is None:
        level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet))
    coeffs = pywt.wavedec(signal, wavelet, level=min(level, 6))
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thresholded_coeffs = [coeffs[0]]
    for coeff in coeffs[1:]:
        lambd = adaptive_threshold(coeff, sigma, alpha)
        thresholded_coeffs.append(soft_threshold(coeff, lambd))
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet)
    return denoised_signal[:len(signal)]  # 确保输出长度与输入一致

def read_signal_data(file_path):
    """读取信号数据，增加异常处理"""
    try:
        with open(file_path, 'r') as file:
            signal = np.array([float(line.strip()) for line in file if line.strip()])
        return signal
    except Exception as e:
        print(f"Error reading data: {e}")
        return None

# 配置文件路径和去噪参数
file_path = r"D:\shudeng\ProofingTool\数据\215711126520129_OriData_20221102055610(1355).ini"
wavelet_base = 'db8'
alpha_value = 2.0

# 从文件读取数据
noisy_signal = read_signal_data(file_path)
if noisy_signal is not None:
    # 小波去噪
    denoised_signal = wavelet_denoising(noisy_signal, wavelet=wavelet_base, alpha=alpha_value)

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
