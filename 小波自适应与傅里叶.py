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

# 自适应阈值计算
def calculate_threshold(coeffs, level, max_level):
    # 在高频层次使用较高的阈值，低频层次使用较低的阈值
    return np.median(np.abs(coeffs)) / (0.6745 * (1 + (level / max_level) ** 2))

# 自适应小波阈值函数
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

# 自适应多级小波去噪函数
def adaptive_multilevel_wavelet_denoising(signal, wavelet_name='db4', alpha_range=(0.8, 1.2)):
    if len(signal) == 0:
        raise ValueError("信号长度为0，无法处理空信号。")

    # 固定分解层数为5
    level = 5  
    max_level = level

    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    thresholded_coeffs = []
    
    # 根据层次应用不同的阈值和不同的alpha值
    for i in range(1, len(coeffs)):
        threshold = calculate_threshold(coeffs[i], i, max_level)  # 根据层次计算阈值
        alpha = np.linspace(alpha_range[0], alpha_range[1], max_level)[i - 1]  # alpha随层次线性变化
        thresholded_coeffs.append(adaptive_wavelet_threshold(coeffs[i], threshold, alpha))
    
    # 低频部分保持不变
    thresholded_coeffs.insert(0, coeffs[0])

    # 小波重构
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet_name)
    return denoised_signal

# 傅里叶变换去噪函数
def fourier_denoising(signal, alpha_range=(0.8, 1.2), max_level=5):
    if len(signal) == 0:
        raise ValueError("信号长度为0，无法处理空信号。")
    
    # 进行傅里叶变换
    fft_coeffs = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal))
    
    # 计算要滤波的频率范围
    cutoff_freq = np.linspace(alpha_range[0], alpha_range[1], max_level)
    
    # 根据频率范围去除高频噪声，假设cutoff_freq的最后一个值为最大的滤波频率
    for i, cutoff in enumerate(cutoff_freq):
        fft_coeffs[np.abs(freqs) > cutoff] = 0  # 低通滤波
    
    # 反变换回时域
    denoised_signal = np.fft.ifft(fft_coeffs).real
    
    return denoised_signal

# 结合小波和傅里叶去噪
def combined_wavelet_fourier_denoising(signal, wavelet_name='db4', alpha_range=(0.8, 1.2), max_level=5):
    # 1. 先进行小波去噪
    wavelet_denoised_signal = adaptive_multilevel_wavelet_denoising(signal, wavelet_name, alpha_range)

    # 2. 对小波去噪后的信号进行傅里叶去噪
    combined_denoised_signal = fourier_denoising(wavelet_denoised_signal, alpha_range, max_level)
    
    return combined_denoised_signal

# 可视化信号与去噪效果，并保存图像
def plot_signal_comparison(original_signal, denoised_signal, file_name, output_folder):
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(original_signal)
    plt.title(f'原始信号 - {file_name}')

    plt.subplot(2, 1, 2)
    plt.plot(denoised_signal)
    plt.title(f'小波+傅里叶联合去噪信号 - {file_name}')

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
            
            # 进行小波和傅里叶联合去噪
            try:
                denoised_signal = combined_wavelet_fourier_denoising(signal)
            except Exception as e:
                print(f"处理 {file_name} 时发生错误: {e}")
                continue
            
            # 可视化并保存图形
            plot_signal_comparison(signal, denoised_signal, file_name.split('.')[0], output_folder)

# 主程序入口
if __name__ == "__main__":
    input_folder = r"C:\Users\86136\Desktop\新建文件夹"  # 输入文件夹路径
    output_folder = r"C:\Users\86136\Desktop\新建文件夹\小波与傅里叶"  # 输出文件夹路径
    
    process_folder(input_folder, output_folder)
