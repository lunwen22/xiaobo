import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import EMD
from scipy.signal import hilbert, savgol_filter

# 读取 .ini 文件数据
def read_ini_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    signal = np.array([float(line.strip()) for line in lines])
    return signal

# 使用 Savitzky-Golay 滤波器进行去噪处理
def denoise_signal(signal, window_length=101, polyorder=3):
    return savgol_filter(signal, window_length, polyorder)

# 进行 EMD 分解
def perform_emd(signal):
    emd = EMD()
    imfs = emd(signal)
    return imfs

# 计算希尔伯特变换后的瞬时频率和瞬时振幅
def hilbert_transform(imfs):
    amplitude_envelope = []
    instantaneous_frequency = []
    
    for imf in imfs:
        analytic_signal = hilbert(imf)
        amplitude_envelope.append(np.abs(analytic_signal))
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency.append(np.diff(instantaneous_phase) / (2.0 * np.pi) * 1000)  # 假设采样率为1000Hz
    
    return amplitude_envelope, instantaneous_frequency

# 绘制结果
def plot_results(signal, imfs, amplitude_envelope, instantaneous_frequency):
    plt.figure(figsize=(12, 12))
    plt.subplot(511)
    plt.plot(signal, 'r')
    plt.title("Original Signal")
    
    plt.subplot(512)
    for i, imf in enumerate(imfs):
        plt.plot(imf)
    plt.title("IMFs")
    
    plt.subplot(513)
    for env in amplitude_envelope:
        plt.plot(env)
    plt.title("Amplitude Envelope")
    
    plt.subplot(514)
    for freq in instantaneous_frequency:
        plt.plot(freq)
    plt.title("Instantaneous Frequency")
    
    plt.tight_layout()
    plt.show()

# 计算信噪比（SNR）
def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# 主函数
def main(file_path):
    original_signal = read_ini_data(file_path)
    denoised_signal = denoise_signal(original_signal)
    imfs = perform_emd(denoised_signal)
    amplitude_envelope, instantaneous_frequency = hilbert_transform(imfs)
    plot_results(original_signal, imfs, amplitude_envelope, instantaneous_frequency)
    
    # 假设噪声为原始信号减去去噪信号
    noise = original_signal - denoised_signal

    # 计算去噪前后的信噪比
    snr_before = calculate_snr(original_signal, noise)
    snr_after = calculate_snr(denoised_signal, noise)

    print(f'SNR before denoising: {snr_before:.2f} dB')
    print(f'SNR after denoising: {snr_after:.2f} dB')

# 示例调用
file_path = 'D:\\shudeng\\ProofingTool\\数据\\141361363626636_OriData_20220801154803.ini'
main(file_path)
