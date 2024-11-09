import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 从ini文件中读取数据，假设每行都是一个数据点
def load_data_from_ini(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    # 将每行的数值转为浮点数，并存入数组
    data = np.array([float(line.strip()) for line in data if line.strip()])
    return data

# 定义自适应小波阈值函数
def adaptive_wavelet_threshold(wavelet_coeffs, threshold, alpha):
    """
    自适应小波阈值函数
    :param wavelet_coeffs: 小波系数
    :param threshold: 阈值
    :param alpha: 调整系数
    :return: 阈值处理后的系数
    """
    result_coeffs = np.sign(wavelet_coeffs) * np.maximum((np.abs(wavelet_coeffs) - threshold), 0) ** alpha
    return result_coeffs

# 基于频率成分调整阈值的函数
def calculate_frequency_adjusted_threshold(signal_length, wavelet_name, level, coeffs):
    """
    根据信号的特定频率成分调整不同层次的阈值
    :param signal_length: 信号长度
    :param wavelet_name: 使用的小波类型
    :param level: 分解层数
    :param coeffs: 小波系数
    :return: 阈值列表
    """
    sampling_rate = 1.0  # 假设采样率为1，你可以根据实际信号修改采样率
    nyquist_freq = sampling_rate / 2.0  # 奈奎斯特频率
    wavelet = pywt.Wavelet(wavelet_name)

    # 计算每个层级对应的频率范围
    threshold_list = []
    for i in range(1, level + 1):
        # 每一层级的小波系数对应的频带
        band_min = nyquist_freq / (2 ** i)
        band_max = nyquist_freq / (2 ** (i - 1))

        # 根据频率带调整阈值（在高频使用较大的阈值，低频较小阈值）
        frequency_adjusted_factor = (band_min + band_max) / nyquist_freq  # 基于频率范围的缩放因子
        threshold = np.median(np.abs(coeffs[i])) / 0.6745 * frequency_adjusted_factor  # 调整后的阈值
        threshold_list.append(threshold)

    return threshold_list

# 自适应小波去噪函数
def adaptive_wavelet_denoising(signal, wavelet_name='db4', alpha_range=(0.8, 1.0)):
    if len(signal) == 0:
        raise ValueError("信号长度为0，无法处理空信号。")

    # 自动计算合适的小波分解层数，防止边界效应
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet_name).dec_len)
    level = min(5, max_level)  # 防止使用过高的分解层数

    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)

    # 基于频率调整不同层次的阈值
    thresholds = calculate_frequency_adjusted_threshold(len(signal), wavelet_name, level, coeffs)

    thresholded_coeffs = []
    
    # 根据不同频率范围应用不同的阈值与alpha值
    for i in range(1, len(coeffs)):  # 跳过低频部分
        threshold = thresholds[i - 1]  # 从阈值列表中提取
        alpha = np.linspace(alpha_range[0], alpha_range[1], len(coeffs) - 1)[i - 1]  # alpha随层次线性变化
        thresholded_coeffs.append(adaptive_wavelet_threshold(coeffs[i], threshold, alpha))
    
    # 保留低频部分
    thresholded_coeffs.insert(0, coeffs[0])

    # 小波重构
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet_name)

    # 使用Savitzky-Golay滤波器对信号进行平滑处理，消除残留的高频噪声
    denoised_signal = savgol_filter(denoised_signal, window_length=21, polyorder=2)

    return denoised_signal

# 可视化信号与去噪效果，并保存图像
def plot_signal_comparison(original_signal, denoised_signal, file_name, output_folder):
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(original_signal)
    plt.title(f'原始信号 - {file_name}')

    plt.subplot(2, 1, 2)
    plt.plot(denoised_signal)
    plt.title(f'自适应小波阈值去噪信号 - {file_name}')

    plt.tight_layout()
    
    # 保存图片到指定文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_path = os.path.join(output_folder, f'{file_name}.png')
    plt.savefig(output_path)
    plt.close()

# 处理文件夹中所有文件
def process_folder(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".ini"):
            file_path = os.path.join(input_folder, file_name)
            
            # 从文件加载信号
            signal = load_data_from_ini(file_path)
            
            if len(signal) == 0:
                print(f"文件 {file_name} 为空，跳过处理。")
                continue
            
            # 进行自适应小波去噪
            try:
                denoised_signal = adaptive_wavelet_denoising(signal)
            except Exception as e:
                print(f"处理 {file_name} 时发生错误: {e}")
                continue
            
            # 可视化并保存图形
            plot_signal_comparison(signal, denoised_signal, file_name.split('.')[0], output_folder)

# 主程序入口
if __name__ == "__main__":
    input_folder = r"C:\Users\86136\Desktop\新建文件夹"  # 输入文件夹路径
    output_folder = r"C:\Users\86136\Desktop\新建文件夹\jieguo6"  # 输出文件夹路径
    
    process_folder(input_folder, output_folder)
