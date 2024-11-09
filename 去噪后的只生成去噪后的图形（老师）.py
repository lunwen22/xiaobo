import os
import numpy as np
import pywt
import matplotlib.pyplot as plt

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

# 使用SNR来选择最优alpha和最优层数的网格搜索
def grid_search_alpha_and_level(signal, original_signal, alpha_range=(0.5, 1.5), level_range=(2, 8), step=0.1):
    best_alpha = alpha_range[0]
    best_level = level_range[0]
    best_snr = -np.inf  # 初始化为负无穷大
    for level in range(level_range[0], level_range[1] + 1):
        for alpha in np.arange(alpha_range[0], alpha_range[1] + step, step):
            denoised_signal = adaptive_wavelet_denoising(signal, alpha=alpha, level=level)
            snr = calculate_snr(original_signal, denoised_signal)
            if snr > best_snr:
                best_snr = snr
                best_alpha = alpha
                best_level = level
    return best_alpha, best_level

# 只生成去噪后的图像并保存
def plot_denoised_signal(denoised_signal, file_name, output_folder):
    plt.figure(figsize=(10, 6))
    plt.plot(denoised_signal)
    plt.title(f'自适应小波阈值去噪信号 - {file_name}')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_path = os.path.join(output_folder, f'{file_name}.png')
    plt.savefig(output_path)
    plt.close()

# 处理文件夹中所有文件，并自动优化alpha和层数
def process_folder(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".ini"):
            file_path = os.path.join(input_folder, file_name)
            signal = load_data_from_ini(file_path)
            if len(signal) == 0:
                print(f"文件 {file_name} 为空，跳过处理。")
                continue
            
            # 将原始信号作为基准（没有去噪之前的信号）
            original_signal = signal.copy()
            
            # 寻找最佳alpha和层数
            best_alpha, best_level = grid_search_alpha_and_level(signal, original_signal)
            print(f"文件 {file_name} 的最佳 alpha 为: {best_alpha}, 最佳层数为: {best_level}")
            
            # 使用最优alpha和层数进行去噪
            denoised_signal = adaptive_wavelet_denoising(signal, alpha=best_alpha, level=best_level)
            
            # 保存去噪后的结果图像
            plot_denoised_signal(denoised_signal, file_name.split('.')[0], output_folder)

# 主程序入口
if __name__ == "__main__":
    input_folder = r"D:\shudeng\ProofingTool\数据"
    output_folder = r"D:\shudeng\ProofingTool\数据\自动优化alpha信噪比1234"
    process_folder(input_folder, output_folder)
