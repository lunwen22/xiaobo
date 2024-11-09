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

# 自适应小波去噪函数
def adaptive_wavelet_denoising(signal, wavelet_name='db4', alpha=1.2):
    if len(signal) == 0:
        raise ValueError("信号长度为0，无法处理空信号。")

    # 自动计算合适的小波分解层数，防止边界效应
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet_name).dec_len)
    level = min(5, max_level)  # 防止使用过高的分解层数

    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    thresholded_coeffs = []
    
    # 根据层数应用不同的阈值
    for i in range(len(coeffs)):
        threshold = np.median(np.abs(coeffs[i])) / 0.6745  # 调整后的阈值计算方法
        thresholded_coeffs.append(adaptive_wavelet_threshold(coeffs[i], threshold, alpha))
    
    # 小波重构
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet_name)
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
    output_folder = r"C:\Users\86136\Desktop\新建文件夹\jieguo4"  # 输出文件夹路径
    
    process_folder(input_folder, output_folder)
