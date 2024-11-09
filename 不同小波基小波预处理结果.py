import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.fftpack import fft, ifft

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 从 ini 文件中加载数据
def load_data_from_ini(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return np.array([float(line.strip()) for line in data if line.strip()])

# 计算信噪比 (SNR)
def calculate_snr(original_signal, denoised_signal):
    noise = original_signal - denoised_signal
    snr = 10 * np.log10(np.sum(original_signal**2) / np.sum(noise**2))
    return snr

# 计算小波熵 (Wavelet Entropy)
def calculate_wavelet_entropy(signal, wavelet='db4', max_level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)
    energies = [np.sum(np.abs(detail)**2) for detail in coeffs[1:]]
    total_energy = np.sum(energies)
    probabilities = [energy / total_energy for energy in energies]
    wavelet_entropy = -np.sum([p * np.log(p) for p in probabilities if p > 0])
    return wavelet_entropy

# 频域滤波 (消除特定频段噪声)
def frequency_domain_filter(signal, low_freq=0, high_freq=None, sample_rate=1.0):
    fft_signal = fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/sample_rate)
    
    # 创建滤波掩码
    if high_freq is None:
        high_freq = np.max(freqs)
    mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
    
    # 应用滤波
    fft_signal[~mask] = 0
    filtered_signal = np.real(ifft(fft_signal))
    return filtered_signal

# 根据信号长度和小波基找到最大分解层数
def find_max_level(signal_length, wavelet):
    return pywt.dwt_max_level(signal_length, wavelet)

# 自适应选择最优小波基
def find_optimal_wavelet(signal):
    wavelet_families = ['db', 'coif', 'sym', 'bior', 'rbio']
    best_snr = -np.inf
    best_wavelet = None
    
    for family in wavelet_families:
        wavelet_list = pywt.wavelist(family)
        for wavelet in wavelet_list:
            max_level = find_max_level(len(signal), wavelet)  # 计算最大分解层数
            try:
                denoised_signal = adaptive_wavelet_denoise(signal, wavelet=wavelet, max_level=max_level)
                snr = calculate_snr(signal, denoised_signal)
                if snr > best_snr:
                    best_snr = snr
                    best_wavelet = wavelet
            except ValueError:
                continue

    print(f"最佳小波基: {best_wavelet}，最佳 SNR: {best_snr}")
    return best_wavelet

# 自适应小波去噪
def adaptive_wavelet_denoise(signal, wavelet='db4', max_level=5, Tr=0.06, increase_factor=2.0):
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)
    details = coeffs[1:]  # 仅保留细节部分

    # 自适应确定最佳分解层数
    k = 0
    for j, detail in enumerate(details):
        Sj = np.max(np.abs(detail)) / np.sum(np.abs(detail))
        if Sj <= Tr:
            k = j
        else:
            break

    # 动态调整阈值并进行去噪处理
    for j in range(1, k + 1):
        detail = details[j - 1]
        mu_j = np.mean(detail)
        delta_j = np.std(detail)
        
        # 根据峰值比自适应调整阈值因子
        Sj = np.max(np.abs(detail)) / np.sum(np.abs(detail))
        if Sj <= 0.01:
            beta_L, beta_H = 1.0 * increase_factor, 1.0 * increase_factor
        elif 0.01 < Sj <= 0.03:
            beta_L, beta_H = 0.9 * increase_factor, 0.9 * increase_factor
        else:
            Sj_L = np.max(detail[detail < 0]) / np.sum(np.abs(detail[detail < 0]))
            Sj_H = np.max(detail[detail > 0]) / np.sum(np.abs(detail[detail > 0]))
            beta_L = (1 - Sj_L / (Sj_L + Sj_H)) * increase_factor
            beta_H = (1 - Sj_H / (Sj_L + Sj_H)) * increase_factor
        
        # 计算上下阈值
        lambda_L = mu_j - beta_L * delta_j
        lambda_H = mu_j + beta_H * delta_j

        # 应用阈值函数
        details[j - 1] = np.where((detail >= lambda_L) & (detail <= lambda_H), 0, detail)
    
    coeffs[1:] = details
    denoised_signal = pywt.waverec(coeffs, wavelet)
    
    # 修正信号长度
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    else:
        denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), mode='constant')
    
    return denoised_signal

# 结合频域滤波和小波去噪
def combined_denoise(signal, sample_rate=1.0, low_freq=0, high_freq=None):
    # 频域滤波预处理
    filtered_signal = frequency_domain_filter(signal, low_freq=low_freq, high_freq=high_freq, sample_rate=sample_rate)
    
    # 最优小波基选择
    best_wavelet = find_optimal_wavelet(filtered_signal)
    
    # 小波去噪
    max_level = find_max_level(len(filtered_signal), best_wavelet)
    denoised_signal = adaptive_wavelet_denoise(filtered_signal, wavelet=best_wavelet, max_level=max_level)
    return denoised_signal

# 绘制信号对比图，并统一纵坐标
def plot_signal_comparison(original_signal, denoised_signal, file_name, output_folder):
    # 计算统一的y轴范围
    y_min = min(np.min(original_signal), np.min(denoised_signal))
    y_max = max(np.max(original_signal), np.max(denoised_signal))

    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(original_signal)
    plt.ylim(y_min, y_max)  # 统一纵坐标范围
    plt.title(f'原始信号 - {file_name}')

    plt.subplot(2, 1, 2)
    plt.plot(denoised_signal)
    plt.ylim(y_min, y_max)  # 统一纵坐标范围
    plt.title(f'联合频域与小波去噪信号 - {file_name}')

    plt.tight_layout()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_path = os.path.join(output_folder, f'{file_name}.png')
    plt.savefig(output_path)
    plt.close()

# 处理文件夹中的所有文件
def process_folder(input_folder, output_folder, sample_rate=1.0, low_freq=0, high_freq=None):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".ini"):
            file_path = os.path.join(input_folder, file_name)
            signal = load_data_from_ini(file_path)
            if len(signal) == 0:
                print(f"文件 {file_name} 为空，跳过处理。")
                continue
            
            original_signal = signal.copy()
            denoised_signal = combined_denoise(signal, sample_rate=sample_rate, low_freq=low_freq, high_freq=high_freq)
            
            # 绘制并保存图像
            plot_signal_comparison(original_signal, denoised_signal, file_name.split('.')[0], output_folder)

# 主程序入口
if __name__ == "__main__":
    input_folder = r"D:\shudeng\boxongtu\log"
    output_folder = r"D:\shudeng\boxongtu\log\联合去噪3"
    process_folder(input_folder, output_folder, sample_rate=1.0, low_freq=0.1, high_freq=20)
