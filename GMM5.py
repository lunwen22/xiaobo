import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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

# 使用手肘法选择最佳 K 值
def elbow_method(features, max_clusters=10):
    sse = []  # 存储不同 K 值下的簇内误差
    k_range = range(1, max_clusters + 1)

    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(features)
        sse.append(-gmm.score(features))  # GMM没有直接的SSE，但负对数似然可以替代

    # 绘制手肘图
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, sse, marker='o')
    plt.xlabel('聚类数量 (K)')
    plt.ylabel('簇内误差（负对数似然）')
    plt.title('手肘法确定最佳聚类数量')
    plt.show()

    # 手动观察图形选择最佳 K 值
    best_k = int(input("根据手肘图选择最佳 K 值："))
    return best_k

# 使用手肘法自动确定的 K 值进行聚类
def perform_clustering_with_elbow(features):
    best_k = elbow_method(features, max_clusters=10)
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

# 自动调整参数函数
def auto_adjust_parameters(cluster_stats):
    max_amplitude = max([stats['avg_amplitude'] for stats in cluster_stats.values()])
    min_amplitude = min([stats['avg_amplitude'] for stats in cluster_stats.values()])
    max_std = max([stats['avg_std_dev'] for stats in cluster_stats.values()])
    
    amplitude_threshold_high = (max_amplitude + min_amplitude) / 2
    amplitude_threshold_mid = amplitude_threshold_high * 0.8
    amplitude_threshold_low = min_amplitude * 1.2
    
    std_threshold_high = max_std * 0.8
    std_threshold_low = max_std * 0.5
    
    return {
        'amplitude_threshold_high': amplitude_threshold_high,
        'amplitude_threshold_mid': amplitude_threshold_mid,
        'amplitude_threshold_low': amplitude_threshold_low,
        'std_threshold_high': std_threshold_high,
        'std_threshold_low': std_threshold_low
    }

# 噪音分类函数
def classify_noise_based_on_adjusted_params(features, labels, adjusted_params):
    noise_types = {}
    
    for cluster_label in np.unique(labels):
        cluster_data = features[labels == cluster_label]
        
        avg_amplitude = np.mean([np.ptp(data) for data in cluster_data])
        avg_std = np.mean([np.std(data) for data in cluster_data])
        
        if avg_amplitude > adjusted_params['amplitude_threshold_high']:
            noise_strength = "强噪音"
        elif avg_amplitude > adjusted_params['amplitude_threshold_mid']:
            noise_strength = "中等噪音"
        else:
            noise_strength = "弱噪音"
        
        if avg_amplitude > adjusted_params['amplitude_threshold_high'] * 1.5:
            noise_level = "高噪音"
        elif avg_amplitude > adjusted_params['amplitude_threshold_mid'] * 1.2:
            noise_level = "中噪音"
        else:
            noise_level = "低噪音"
        
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

# 可视化聚类结果
def plot_clusters(features, labels, gmm, noise_classification):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    centers = pca.transform(gmm.means_)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='簇中心')

    for i, (x, y) in enumerate(centers):
        noise_type = noise_classification.get(i, {}).get('噪音类型', '未知')
        noise_strength = noise_classification.get(i, {}).get('噪音强度', '未知')
        noise_level = noise_classification.get(i, {}).get('噪音级别', '未知')
        plt.text(x, y, f'{noise_type}\n{noise_strength}\n{noise_level}', fontsize=10, ha='center', color='black',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='black', boxstyle='round,pad=0.5'))

    plt.title('聚类结果')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(scatter, label='聚类标签')
    plt.legend()
    plt.show()

# 保存每个聚类中的图像
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
    features = [extract_signal_features(img) for img in images]
    features = np.array(features)

    labels, gmm = perform_clustering_with_elbow(features)
    save_clustered_images_by_label(image_folder, file_names, labels, output_folder)

    cluster_stats = calculate_cluster_stats(features, labels)
    adjusted_params = auto_adjust_parameters(cluster_stats)

    noise_classification = classify_noise_based_on_adjusted_params(features, labels, adjusted_params)
    plot_clusters(features, labels, gmm, noise_classification)

    for cluster_label, classification in noise_classification.items():
        print(f"簇 {cluster_label}: {classification['噪音强度']} - {classification['噪音类型']} - {classification['噪音级别']}")

if __name__ == "__main__":
    input_folder = r"D:\shudeng\波形图\结果1"
    output_folder = r"D:\shudeng\波形图\结果1\手肘法_gmm_分类结合结果5"
    process_images_with_combined_method(input_folder, output_folder)
