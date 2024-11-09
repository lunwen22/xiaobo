import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.signal import welch, find_peaks
from PIL import Image
from scipy.stats import skew, kurtosis

# 设置中文字体支持
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei（黑体）
matplotlib.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 读取图像并转换为特征向量
def load_images_as_features(image_folder):
    images = []
    file_names = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_folder, file_name)
            img = Image.open(file_path).convert('L')
            img = img.resize((128, 128))
            img_data = np.asarray(img).flatten
            file_names.append(file_name)
    return np.array(images), file_names  # 转换为NumPy数组

# 提取频域特征 (如频率峰值，总能量，峰值数量)
def extract_frequency_features(signal):
    signal_length = len(signal)
    nperseg = min(256, signal_length // 2) if signal_length > 2 else 1
    if signal_length > 2:
        freqs, psd = welch(signal, nperseg=nperseg)
        peak_freq = freqs[np.argmax(psd)]  # 频谱峰值
        total_energy = np.sum(psd)  # 总能量
        peak_count = len(find_peaks(psd)[0])  # 频域峰值的数量
    else:
        peak_freq = 0
        total_energy = 0
        peak_count = 0
    return peak_freq, total_energy, peak_count

# 提取时域特征 (如标准差，均值，RMS，偏度，峰度)
def extract_time_features(signal):
    std = np.std(signal)
    mean = np.mean(signal)
    peak_to_peak = np.ptp(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    rms = np.sqrt(np.mean(signal ** 2))
    return [std, mean, peak_to_peak, skewness, kurt, rms]

# 添加自相关特征
def extract_autocorrelation(signal, lag=1):
    return np.corrcoef(signal[:-lag], signal[lag:])[0, 1]

# 提取时域和频域特征的组合
def extract_features(signal):
    time_features = extract_time_features(signal)
    peak_freq, total_energy, peak_count = extract_frequency_features(signal)
    autocorr_1 = extract_autocorrelation(signal, lag=1)
    autocorr_2 = extract_autocorrelation(signal, lag=2)
    return time_features + [total_energy, peak_freq, peak_count, autocorr_1, autocorr_2]

# 根据特征命名噪音类型
def assign_cluster_names(features, labels):
    cluster_names = {}
    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # -1 代表噪声点
            cluster_names[cluster_id] = "噪声"
            continue
        cluster_features = np.mean([features[i] for i in range(len(features)) if labels[i] == cluster_id], axis=0)
        peak_freq = cluster_features[-4]  # 频率特征
        total_energy = cluster_features[-5]  # 总能量特征
        rms_value = cluster_features[5]  # RMS值（第6个特征）
        
        # 根据频率特性命名
        if peak_freq < 200:
            name = "低频噪音"
        elif peak_freq > 1000:
            name = "高频噪音"
        else:
            name = "中频噪音"
        
        # 根据能量和RMS命名
        if total_energy > 500 and rms_value > 0.5:
            name = f"强{name}"
        else:
            name = f"弱{name}"
        
        cluster_names[cluster_id] = name
    return cluster_names

# 使用DBSCAN聚类
def perform_dbscan_clustering(features, eps=0.5, min_samples=5):
    features = StandardScaler().fit_transform(features)  # 标准化特征
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)
    return labels

# 可视化聚类结果
def plot_and_save_results(reduced_features, labels, cluster_names, output_folder):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=100, alpha=0.7)
    plt.title('降维后的噪音类型聚类结果')
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')
    plt.colorbar(label='簇')

    for cluster_id in np.unique(labels):
        if cluster_id in cluster_names:
            plt.text(reduced_features[labels == cluster_id, 0].mean(),
                     reduced_features[labels == cluster_id, 1].mean(),
                     cluster_names[cluster_id],
                     fontsize=12, weight='bold',
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='black', boxstyle='round,pad=0.5'))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, 'dbscan_clustering_result.png')
    plt.savefig(output_path)
    plt.show()

# 主程序入口
def process_images_with_dbscan(image_folder, output_folder, eps=0.5, min_samples=5):
    images, file_names = load_images_as_features(image_folder)
    features = [extract_features(img) for img in images]

    # 使用DBSCAN进行聚类
    labels = perform_dbscan_clustering(features, eps=eps, min_samples=min_samples)

    # PCA降维
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(np.array(features))

    # 分配簇名
    cluster_names = assign_cluster_names(features, labels)

    # 可视化并保存结果
    plot_and_save_results(reduced_features, labels, cluster_names, output_folder)

if __name__ == "__main__":
    input_folder = r"D:\shudeng\波形图\结果1"
    output_folder = r"D:\shudeng\波形图\结果1\DBSCAN聚类结果"
    
    process_images_with_dbscan(input_folder, output_folder, eps=0.5, min_samples=5)
