import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

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

# 定义均方误差 (MSE) 计算函数
def calculate_mse(original_signal, denoised_signal):
    return np.mean((original_signal - denoised_signal) ** 2)

# 定义SSIM计算函数
def calculate_ssim(original_signal, denoised_signal):
    data_range = np.max(original_signal) - np.min(original_signal)
    return ssim(original_signal, denoised_signal, data_range=data_range)

# 改进的自适应阈值函数，带有无偏估计
def improved_wavelet_threshold(wavelet_coeffs, threshold):
    result_coeffs = np.zeros_like(wavelet_coeffs)
    for i in range(len(wavelet_coeffs)):
        coeff = wavelet_coeffs[i]
        abs_coeff = np.abs(coeff)
        
        if abs_coeff < threshold:
            result_coeffs[i] = 0
        elif abs_coeff == threshold:
            factor = (2 / np.pi) * np.arctan(abs_coeff - threshold)
            result_coeffs[i] = factor * coeff + (1 - factor) * np.sign(coeff)
        else:
            factor = (1 + (abs_coeff**2 - (2 + abs_coeff * threshold)) / (1 + abs_coeff**2)) * \
                     np.log2(1 + abs_coeff) / (abs_coeff - threshold + 1)
            result_coeffs[i] = factor * coeff
    
    return result_coeffs

# 自适应小波去噪函数
def adaptive_wavelet_denoising(signal, wavelet_name, level=5, threshold=0.5):
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    thresholded_coeffs = []
    
    for i in range(len(coeffs)):
        thresholded_coeffs.append(improved_wavelet_threshold(coeffs[i], threshold))
    
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet_name)
    
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    else:
        denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), mode='constant')
    
    return denoised_signal

# 更新网格搜索范围
def grid_search_best_params(signal, original_signal, level=5, threshold_range=(0.05, 1.0), threshold_step=0.05):
    best_wavelet = None
    best_threshold = threshold_range[0]
    best_snr = -np.inf
    best_denoised_signal = None
    
    # 优化小波基选择
    wavelet_list = ['db2', 'db4', 'coif1', 'sym2']  # 可以根据实验选择更适合的基

    for wavelet_name in wavelet_list:
        for threshold in np.arange(threshold_range[0], threshold_range[1], threshold_step):
            denoised_signal = adaptive_wavelet_denoising(signal, wavelet_name, level, threshold)
            snr = calculate_snr(original_signal, denoised_signal)
            
            print(f"Testing wavelet: {wavelet_name}, threshold: {threshold}, SNR: {snr}")
            
            if snr > best_snr:
                best_snr = snr
                best_wavelet = wavelet_name
                best_threshold = threshold
                best_denoised_signal = denoised_signal

    return best_wavelet, best_threshold, best_snr, best_denoised_signal


# 可视化信号与去噪效果，并保存图像
def plot_signal_comparison(original_signal, denoised_signal, file_name, output_folder):
    plt.figure(figsize=(10, 6))
    
    # 获取原始信号的纵坐标范围
    y_min, y_max = np.min(original_signal), np.max(original_signal)
    
    plt.subplot(2, 1, 1)
    plt.plot(original_signal)
    plt.ylim(y_min, y_max)  # 设置与原始信号相同的纵坐标范围
    plt.title(f'原始信号 - {file_name}')

    plt.subplot(2, 1, 2)
    plt.plot(denoised_signal)
    plt.ylim(y_min, y_max)  # 设置与原始信号相同的纵坐标范围
    plt.title(f'自适应小波阈值去噪信号 - {file_name}')

    plt.tight_layout()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_path = os.path.join(output_folder, f'{file_name}.png')
    plt.savefig(output_path)
    plt.close()


# 处理文件夹中所有文件，并自动优化小波基和阈值
def process_folder(input_folder, output_folder, threshold_range=(0.1, 2.0), threshold_step=0.1):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".ini"):
            file_path = os.path.join(input_folder, file_name)
            signal = load_data_from_ini(file_path)
            if len(signal) == 0:
                print(f"文件 {file_name} 为空，跳过处理。")
                continue
            
            original_signal = signal.copy()
            
            # 网格搜索最佳小波基和阈值
            best_wavelet, best_threshold, best_snr, best_denoised_signal = grid_search_best_params(
                signal, original_signal, threshold_range=threshold_range, threshold_step=threshold_step)
            
            print(f"文件 {file_name} 的最佳小波基为: {best_wavelet}, 最佳阈值为: {best_threshold}, SNR: {best_snr}")
            
            # 保存去噪后的结果图像
            plot_signal_comparison(original_signal, best_denoised_signal, file_name.split('.')[0], output_folder)

# 主程序入口
if __name__ == "__main__":
    input_folder = r"D:\shudeng\boxongtu\log"
    output_folder = r"D:\shudeng\boxongtu\不同小波基小波预处理结果3"
    process_folder(input_folder, output_folder, threshold_range=(0.1, 2.0), threshold_step=0.1)
