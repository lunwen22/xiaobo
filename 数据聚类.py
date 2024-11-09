import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

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
        threshold = np.median(np.abs(coeffs[i])) / 0.6745  # 这里是阈值的自动定义
        thresholded_coeffs.append(adaptive_wavelet_threshold(coeffs[i], threshold, alpha))
    
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet_name)
    
    # 修正长度以匹配原始信号
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    else:
        denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), mode='constant')
    
    return denoised_signal

# 提取特征进行聚类：这里可以提取频域特征或时域特征
def extract_features(signal):
    # 提取基本的统计特征，如均值、标准差、最大值和最小值
    mean = np.mean(signal)
    std = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    
    # 可以扩展更多特征
    return np.array([mean, std, max_val, min_val])

# 自动确定最优聚类数量
def find_optimal_clusters(features, max_clusters=10):
    sse = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
        sse.append(kmeans.inertia_)
    
    # 通过肘部法找拐点
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), sse, marker='o')
    plt.title('肘部法确定最佳聚类数量')
    plt.xlabel('聚类数量')
    plt.ylabel('SSE (误差平方和)')
    plt.show()

    # 返回用户手动选择的聚类数量
    return int(input("根据肘部法图形选择最佳聚类数量: "))

# 使用 K-Means 聚类噪音类型
def kmeans_clustering(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    return kmeans.labels_

# 可视化聚类结果，将聚类特征绘制成不同颜色的团簇
def plot_clustered_features(features, labels, output_file):
    # 使用PCA将数据降维到2D以便可视化
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='tab10', s=50)
    plt.title('不同颜色的噪音团簇')
    plt.colorbar()
    
    # 保存图像
    plt.savefig(output_file)
    plt.show()

# 处理文件夹中的所有文件
def process_folder(input_folder, output_folder):
    features_list = []
    file_names = []
    
    # Step 1: 处理所有文件，提取信号特征
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".ini"):
            file_path = os.path.join(input_folder, file_name)
            signal = load_data_from_ini(file_path)
            if len(signal) == 0:
                print(f"文件 {file_name} 为空，跳过处理。")
                continue
            
            # 将原始信号作为基准（没有去噪之前的信号）
            original_signal = signal.copy()
            
            # 进行去噪
            denoised_signal = adaptive_wavelet_denoising(signal)
            
            # 提取去噪后的特征
            features = extract_features(denoised_signal)
            features_list.append(features)
            file_names.append(file_name)
    
    # Step 2: 转换为NumPy数组以便于处理
    features_array = np.array(features_list)

    # Step 3: 自动确定聚类数量
    optimal_clusters = find_optimal_clusters(features_array, max_clusters=10)
    
    # Step 4: 使用K-means进行聚类
    noise_labels = kmeans_clustering(features_array, optimal_clusters)
    
    # Step 5: 可视化聚类结果
    plot_clustered_features(features_array, noise_labels, os.path.join(output_folder, '聚类结果图.png'))

# 主程序入口
if __name__ == "__main__":
    input_folder = r"D:\shudeng\ProofingTool\数据\自动优化alpha信噪比1234"
    output_folder = r"D:\shudeng\ProofingTool\数据\自动优化alpha信噪比julei "
    process_folder(input_folder, output_folder)
