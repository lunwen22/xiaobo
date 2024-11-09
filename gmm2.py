import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
from PIL import Image

# 设置中文字体支持
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取图像并转换为特征向量
def load_images_as_features(image_folder):
    images = []
    file_names = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_folder, file_name)
            img = Image.open(file_path).convert('L')
            img = img.resize((64, 64))
            img_data = np.asarray(img).flatten() / 255.0
            images.append(img_data)
            file_names.append(file_name)
    return np.array(images), file_names

# 提取信号的主要特征
def extract_signal_features(signal):
    amplitude = np.ptp(signal)  # 振幅
    std_dev = np.std(signal)    # 标准差
    energy = np.sum(signal**2)  # 能量
    peaks = len(find_peaks(signal)[0])  # 峰值计数
    return amplitude, std_dev, energy, peaks

# 使用 BIC 和 AIC 选择最佳 K 值
def bic_aic_method(features, max_clusters=10):
    bic_scores = []
    aic_scores = []
    k_range = range(1, max_clusters + 1)

    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(features)
        bic_scores.append(gmm.bic(features))
        aic_scores.append(gmm.aic(features))

    # 绘制 BIC 和 AIC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, bic_scores, label='BIC', marker='o')
    plt.plot(k_range, aic_scores, label='AIC', marker='o')
    plt.xlabel('聚类数量 (K)')
    plt.ylabel('准则值')
    plt.title('BIC 和 AIC 确定最佳聚类数量')
    plt.legend()
    plt.show()

    # 返回 BIC 和 AIC 最小值对应的 K 值
    best_k_bic = np.argmin(bic_scores) + 1
    best_k_aic = np.argmin(aic_scores) + 1

    print(f"根据 BIC，最佳 K 值为: {best_k_bic}")
    print(f"根据 AIC，最佳 K 值为: {best_k_aic}")

    return best_k_bic, best_k_aic

# 使用 BIC 或 AIC 自动确定的 K 值进行聚类
def perform_clustering_with_bic_aic(features):
    best_k_bic, best_k_aic = bic_aic_method(features, max_clusters=10)
    
    # 选择使用 BIC 或 AIC 的最佳 K 值，这里以 BIC 为主
    best_k = best_k_bic  # 或者可以改为 best_k_aic

    gmm = GaussianMixture(n_components=best_k, random_state=42)
    labels = gmm.fit_predict(features)
    return labels, gmm

# 计算簇内特征的统计信息
def calculate_cluster_stats(features, labels):
    cluster_stats = {}
    for cluster_label in np.unique(labels):
        cluster_data = features[labels == cluster_label]
        
        # 提取簇内的特征统计信息
        amplitudes = [np.ptp(data) for data in cluster_data]
        std_devs = [np.std(data) for data in cluster_data]
        energies = [np.sum(data**2) for data in cluster_data]
        peaks_count = [len(find_peaks(data)[0]) for data in cluster_data]
        
        cluster_stats[cluster_label] = {
            'avg_amplitude': np.mean(amplitudes),
            'avg_std_dev': np.mean(std_devs),
            'avg_energy': np.mean(energies),
            'avg_peaks': np.mean(peaks_count),
            'max_amplitude': np.max(amplitudes),
            'min_amplitude': np.min(amplitudes)
        }
    return cluster_stats

# 合并相似簇的逻辑
def merge_similar_clusters(cluster_stats, similarity_threshold=0.1):
    merged_clusters = {}
    visited = set()

    for cluster_label, stats in cluster_stats.items():
        if cluster_label in visited:
            continue
        
        merged_clusters[cluster_label] = [cluster_label]
        visited.add(cluster_label)

        for other_label, other_stats in cluster_stats.items():
            if other_label == cluster_label or other_label in visited:
                continue

            # 判断两个簇是否相似
            amplitude_diff = abs(stats['avg_amplitude'] - other_stats['avg_amplitude'])
            std_diff = abs(stats['avg_std_dev'] - other_stats['avg_std_dev'])

            # 如果振幅和标准差差异都小于阈值，则认为它们相似，合并
            if amplitude_diff < similarity_threshold and std_diff < similarity_threshold:
                merged_clusters[cluster_label].append(other_label)
                visited.add(other_label)

    return merged_clusters

# 自动调整参数函数
def auto_adjust_parameters(cluster_stats):
    # 基于每个簇的统计信息动态调整参数
    max_amplitude = max([stats['avg_amplitude'] for stats in cluster_stats.values()])
    min_amplitude = min([stats['avg_amplitude'] for stats in cluster_stats.values()])
    max_std = max([stats['avg_std_dev'] for stats in cluster_stats.values()])
    
    # 动态设置振幅阈值
    amplitude_threshold_high = (max_amplitude + min_amplitude) / 2
    amplitude_threshold_mid = amplitude_threshold_high * 0.8
    amplitude_threshold_low = min_amplitude * 1.2
    
    # 动态设置标准差阈值
    std_threshold_high = max_std * 0.8
    std_threshold_low = max_std * 0.5
    
    return {
        'amplitude_threshold_high': amplitude_threshold_high,
        'amplitude_threshold_mid': amplitude_threshold_mid,
        'amplitude_threshold_low': amplitude_threshold_low,
        'std_threshold_high': std_threshold_high,
        'std_threshold_low': std_threshold_low
    }

# 噪音分类函数，根据自动调整后的参数划分噪音类型
def classify_noise_with_merged_clusters(features, labels, adjusted_params, merged_clusters):
    noise_types = {}
    
    for cluster_label, merged_labels in merged_clusters.items():
        cluster_data = features[np.isin(labels, merged_labels)]  # 合并后的簇数据
        
        # 提取合并后的主要特征
        avg_amplitude = np.mean([np.ptp(data) for data in cluster_data])
        avg_std = np.mean([np.std(data) for data in cluster_data])
        
        # 根据自动调整的阈值进行噪音分类
        if avg_amplitude > adjusted_params['amplitude_threshold_high']:
            noise_strength = "强噪音"
        elif avg_amplitude > adjusted_params['amplitude_threshold_mid']:
            noise_strength = "中等噪音"
        else:
            noise_strength = "弱噪音"
        
        # 根据振幅进一步划分高中低噪音
        if avg_amplitude > adjusted_params['amplitude_threshold_high'] * 1.5:
            noise_level = "高噪音"
        elif avg_amplitude > adjusted_params['amplitude_threshold_mid'] * 1.2:
            noise_level = "中噪音"
        else:
            noise_level = "低噪音"
        
        # 噪音类型判断
        if avg_std > adjusted_params['std_threshold_high']:
            noise_type = "机械振动噪音"
        elif avg_amplitude > adjusted_params['amplitude_threshold_high'] and avg_std < adjusted_params['std_threshold_low']:
            noise_type = "脉冲噪音"
        else:
            noise_type = "随机噪音"
        
        noise_types[cluster_label] = {
            '噪音强度': noise_strength,
            '噪音类型': noise_type,
            '噪音级别': noise_level
        }
    
    return noise_types

# 可视化聚类结果，展示每个簇的中心以及噪音类型
def plot_clusters(features, labels, gmm, merged_clusters, noise_classification):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    centers = pca.transform(gmm.means_)  # 获取簇中心点在PCA后的坐标

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=50)

    # 绘制聚类中心
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='簇中心')

    # 在每个簇中心标注噪音类型
    for i, (x, y) in enumerate(centers):
        noise_type = noise_classification.get(i, {}).get('噪音类型', '未知')
        noise_strength = noise_classification.get(i, {}).get('噪音强度', '未知')
        noise_level = noise_classification.get(i, {}).get('噪音级别', '未知')
        plt.text(x, y, f'{noise_type}\n{noise_strength}\n{noise_level}', fontsize=10, ha='center', color='black',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='black', boxstyle='round,pad=0.5'))

    plt.title('聚类结果（合并相似簇后）')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(scatter, label='聚类标签')
    plt.legend()
    plt.show()

# 保存每个聚类中的图像到对应的文件夹
def save_clustered_images_by_label(image_folder, file_names, labels, output_folder):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_folder = os.path.join(output_folder, f'cluster_{label}')
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)

    for file_name, label in zip(file_names, labels):
        src_image_path = os.path.join(image_folder, file_name)
        dest_image_path = os.path.join(output_folder, f'cluster_{label}', file_name)
        img = Image.open(src_image_path)
        img.save(dest_image_path)

    print(f"所有图像已按聚类结果保存到各自的文件夹中：{output_folder}")

# 主程序入口
def process_images_with_combined_method(image_folder, output_folder):
    images, file_names = load_images_as_features(image_folder)
    
    # 提取特征
    features = [extract_signal_features(img) for img in images]
    features = np.array(features)  # 确保 features 是一个 NumPy 数组

    # 使用 BIC 或 AIC 自动选择最佳 K 值进行聚类
    labels, gmm = perform_clustering_with_bic_aic(features)

    # 保存每个聚类中的图像到对应的文件夹
    save_clustered_images_by_label(image_folder, file_names, labels, output_folder)

    # 计算簇内的统计信息并调整参数
    cluster_stats = calculate_cluster_stats(features, labels)
    adjusted_params = auto_adjust_parameters(cluster_stats)

    # 合并相似簇
    merged_clusters = merge_similar_clusters(cluster_stats)

    # 自动分类噪音类型
    noise_classification = classify_noise_with_merged_clusters(features, labels, adjusted_params, merged_clusters)
    
    # 可视化聚类结果并显示合并后的噪音类型
    plot_clusters(features, labels, gmm, merged_clusters, noise_classification)

    # 输出噪音分类结果
    for cluster_label, classification in noise_classification.items():
        print(f"簇 {cluster_label}: {classification['噪音强度']} - {classification['噪音类型']} - {classification['噪音级别']}")

if __name__ == "__main__":
    input_folder = r"D:\shudeng\波形图\结果1"
    output_folder = r"D:\shudeng\波形图\结果1\BIC_AIC_gmm_分类结合结果3"

    process_images_with_combined_method(input_folder, output_folder)
