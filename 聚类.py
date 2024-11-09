import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 从ini文件中读取数据，假设每行都是一个数据点
def load_data_from_ini(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    data = np.array([float(line.strip()) for line in data if line.strip()])
    return data

# 定义自适应小波阈值函数
def adaptive_wavelet_threshold(wavelet_coeffs, threshold, alpha):
    """应用自适应阈值对小波系数进行处理"""
    result_coeffs = np.sign(wavelet_coeffs) * np.maximum((np.abs(wavelet_coeffs) - threshold), 0) ** alpha
    return result_coeffs

# 自适应小波去噪函数
def adaptive_wavelet_denoising(signal, wavelet_name='coif5', alpha=1.0, level=5):
    """对信号进行自适应小波去噪"""
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    thresholded_coeffs = []
    
    for i in range(len(coeffs)):
        threshold = np.median(np.abs(coeffs[i])) / 0.6745  # 计算自适应阈值
        thresholded_coeffs.append(adaptive_wavelet_threshold(coeffs[i], threshold, alpha))  # 应用阈值化
    
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet_name)  # 重构去噪后的信号
    
    # 修正长度以匹配原始信号
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    else:
        denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), mode='constant')
    
    return denoised_signal

# 对齐信号长度（通过填充零值或裁剪）
def align_signals_length(signals):
    """对齐所有信号的长度"""
    max_len = max(len(signal) for signal in signals)
    aligned_signals = [np.pad(signal, (0, max_len - len(signal)), mode='constant') for signal in signals]
    return np.array(aligned_signals)

# 使用K-Means聚类，并可视化结果
def perform_kmeans_clustering(signals, n_clusters=3):
    """使用K-Means对信号进行聚类"""
    aligned_signals = align_signals_length(signals)  # 先对齐所有信号的长度
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    signals_flattened = aligned_signals  # 已经对齐，所以直接用矩阵形式
    labels = kmeans.fit_predict(signals_flattened)  # 对信号进行聚类
    return labels

# 可视化信号与聚类结果，并保存图像
def plot_signal_comparison_with_clusters(signals, labels, file_names, output_folder):
    """将信号与聚类结果可视化"""
    plt.figure(figsize=(10, 2 * len(signals)))  # 调整figsize以适应所有信号
    num_signals = len(signals)
    
    for i, signal in enumerate(signals):
        plt.subplot(num_signals, 1, i + 1)
        plt.plot(signal, label=f'信号 - {file_names[i]}', color=f'C{labels[i]}')
        plt.title(f'聚类类别: {labels[i]}')
    
    # 调整子图之间的垂直间距
    plt.subplots_adjust(hspace=0.5)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, 'clustered_signals.png')
    plt.savefig(output_path)
    plt.close()

# 主程序入口
def process_folder_with_clustering(input_folder, output_folder, n_clusters=3):
    """处理输入文件夹中的所有信号文件，并进行K-Means聚类"""
    signals = []
    file_names = []
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".ini"):
            file_path = os.path.join(input_folder, file_name)
            signal = load_data_from_ini(file_path)
            if len(signal) == 0:
                print(f"文件 {file_name} 为空，跳过处理。")
                continue

            # 自适应小波去噪
            denoised_signal = adaptive_wavelet_denoising(signal)
            
            signals.append(denoised_signal)
            file_names.append(file_name.split('.')[0])
    
    if len(signals) > 0:
        # 执行K-Means聚类
        labels = perform_kmeans_clustering(signals, n_clusters=n_clusters)
        
        # 可视化聚类结果
        plot_signal_comparison_with_clusters(signals, labels, file_names, output_folder)

# 使用示例
if __name__ == "__main__":
    input_folder = r"D:\shudeng\ProofingTool\数据"
    output_folder = r"D:\shudeng\ProofingTool\数据\聚类分析结果"
    process_folder_with_clustering(input_folder, output_folder)
