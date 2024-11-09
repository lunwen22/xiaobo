import os
import numpy as np
import pywt
from datetime import datetime

# 从ini文件中读取数据，假设每行都是一个数据点
def load_data_from_ini(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    data = np.array([float(line.strip()) for line in data if line.strip()])
    return data

# 定义SNR计算函数
def calculate_snr(original_signal, denoised_signal):
    noise = original_signal - denoised_signal
    snr = 10 * np.log10(np.sum(original_signal**2) / np.sum(noise**2))
    return snr

# 定义自适应小波阈值函数
def adaptive_wavelet_threshold(wavelet_coeffs, threshold, alpha):
    result_coeffs = np.sign(wavelet_coeffs) * np.maximum((np.abs(wavelet_coeffs) - threshold), 0) ** alpha
    return result_coeffs

# 自适应小波去噪函数
def adaptive_wavelet_denoising(signal, wavelet_name='coif5', alpha=1.0, level=5):
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    thresholded_coeffs = []
    
    for i in range(len(coeffs)):
        threshold = np.median(np.abs(coeffs[i])) / 0.6745
        thresholded_coeffs.append(adaptive_wavelet_threshold(coeffs[i], threshold, alpha))
    
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet_name)
    
    # 修正长度以匹配原始信号
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    else:
        denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), mode='constant')
    
    return denoised_signal

# 对不同的小波基进行实验比较
def experiment_with_different_wavelets(signal, original_signal, wavelets, alpha=1.0, level=5):
    best_snr = -np.inf
    best_wavelet = None
    
    for wavelet in wavelets:
        # 进行去噪
        denoised_signal = adaptive_wavelet_denoising(signal, wavelet_name=wavelet, alpha=alpha, level=level)
        # 计算SNR
        snr = calculate_snr(original_signal, denoised_signal)
        print(f"小波基: {wavelet}, SNR: {snr:.2f}")
        
        # 找到最佳的小波基
        if snr > best_snr:
            best_snr = snr
            best_wavelet = wavelet
    
    print(f"最佳小波基: {best_wavelet}, 最佳 SNR: {best_snr:.2f}")
    return best_wavelet, best_snr

# 处理文件夹中所有文件，使用不同小波基，选择最佳小波基并计算SNR
def process_folder(input_folder, output_folder):
    total_original_signal = np.array([])  # 存储所有文件的原始信号
    total_denoised_signal = np.array([])  # 存储所有文件的去噪信号

    # 定义要比较的小波基列表
    wavelet_list = ['coif5', 'db4', 'sym8', 'bior3.5', 'haar']  # 可根据需求扩展

    # 创建输出文件夹，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 创建结果文件
    result_file = os.path.join(output_folder, "SNR_Results.txt")
    
    with open(result_file, 'w') as f:
        f.write("文件处理结果:\n")
    
        for file_name in os.listdir(input_folder):
            if file_name.endswith(".ini"):
                file_path = os.path.join(input_folder, file_name)
                signal = load_data_from_ini(file_path)
                if len(signal) == 0:
                    print(f"文件 {file_name} 为空，跳过处理。")
                    f.write(f"文件 {file_name} 为空，跳过处理。\n")
                    continue
                
                # 将原始信号作为基准（没有去噪之前的信号）
                original_signal = signal.copy()

                # 进行实验，选择最佳的小波基
                best_wavelet, best_snr = experiment_with_different_wavelets(signal, original_signal, wavelets=wavelet_list)
                
                # 写入文件处理信息
                f.write(f"文件: {file_name}, 最佳小波基: {best_wavelet}, 最佳 SNR: {best_snr:.2f}\n")
                
                # 使用最佳小波基进行去噪
                denoised_signal = adaptive_wavelet_denoising(signal, wavelet_name=best_wavelet)

                # 拼接每个文件的原始信号和去噪信号
                total_original_signal = np.concatenate((total_original_signal, original_signal))
                total_denoised_signal = np.concatenate((total_denoised_signal, denoised_signal))

        # 计算总SNR
        total_snr = calculate_snr(total_original_signal, total_denoised_signal)
        print(f"总SNR: {total_snr:.2f}")
        
        # 将总SNR结果写入文件
        f.write(f"\n总SNR: {total_snr:.2f}\n")

# 主程序入口
if __name__ == "__main__":
    input_folder = r"D:\shudeng\boxongtu\log"
    
    # 自动生成的文件夹名称，包含时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = os.path.join(input_folder, f"计算结果_{timestamp}")
    
    process_folder(input_folder, output_folder)
