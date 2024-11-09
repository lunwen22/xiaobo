import os
import numpy as np
from scipy.signal import welch, find_peaks
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from PIL import Image
import librosa
import shutil  # 用于文件复制
import matplotlib.pyplot as plt

# 设置中文字体支持
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 噪声类型分类
def classify_noise_type(low_freq_energy, mid_freq_energy, high_freq_energy, low_freq_threshold, mid_freq_threshold, high_freq_threshold):
    if high_freq_energy > high_freq_threshold:
        return "高频噪音"
    elif mid_freq_energy > mid_freq_threshold:
        return "中频噪音"
    else:
        return "低频噪音"

# 读取图像并转换为特征向量，同时保留原始路径
def load_images_from_folders(folders):
    images = []
    file_names = []
    original_folders = []
    for folder in folders:
        for file_name in os.listdir(folder):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(folder, file_name)
                img = Image.open(file_path).convert('L')  # 将图像转换为灰度
                img = img.resize((64, 64))  # 假设图像统一大小为 64x64
                img_data = np.asarray(img).flatten() / 255.0  # 将图像像素值归一化
                images.append(img_data)
                file_names.append(file_name)  # 保存文件名
                original_folders.append(folder)  # 保存文件所在的原始文件夹
    return np.array(images), file_names, original_folders

# 提取信号特征
def extract_signal_features(signal, fs=1000):
    amplitude = np.ptp(signal)  # 振幅
    std_dev = np.std(signal)    # 标准差
    energy = np.sum(signal**2)  # 能量
    peaks = len(find_peaks(signal)[0])  # 峰值计数

    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    low_freq_energy = np.mean(psd[(freqs >= 0) & (freqs <= 100)])  # 低频：0-100Hz
    mid_freq_energy = np.mean(psd[(freqs > 100) & (freqs <= 300)])  # 中频：100-300Hz
    high_freq_energy = np.mean(psd[freqs > 300])  # 高频：300Hz 以上

    return [amplitude, std_dev, energy, peaks, low_freq_energy, mid_freq_energy, high_freq_energy]

# 自适应计算频率阈值，确保每个频段都有噪音分类，基于ZCR调整
def calculate_adaptive_frequency_thresholds(signal, fs=1000):
    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    
    # 计算ZCR并用于动态调整阈值
    zcr = np.mean(librosa.feature.zero_crossing_rate(signal)[0])

    # 动态调整的频率阈值，根据ZCR调节敏感度
    low_freq_threshold = np.mean(psd[(freqs >= 0) & (freqs <= 100)]) * (1 + zcr)
    mid_freq_threshold = np.mean(psd[(freqs > 100) & (freqs <= 300)]) * (1 + zcr)
    high_freq_threshold = np.mean(psd[freqs > 300]) * (1 + zcr)

    return low_freq_threshold, mid_freq_threshold, high_freq_threshold

# BIC 自动确定最佳 K 值进行聚类
def perform_clustering_with_bic(features, max_clusters=10):
    bic_scores = []
    gmm_models = []
    
    for k in range(1, max_clusters + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(features)
        bic_scores.append(gmm.bic(features))
        gmm_models.append(gmm)

    best_k = np.argmin(bic_scores) + 1
    best_gmm = gmm_models[np.argmin(bic_scores)]
    print(f"根据 BIC，最佳聚类数 (K) 为: {best_k}")
    
    labels = best_gmm.fit_predict(features)
    return labels, best_gmm

# 给每个类别打标签并保存图片
def label_and_save_clusters(features, labels, file_names, original_folders, low_freq_threshold, mid_freq_threshold, high_freq_threshold, output_base_folder):
    for i, label in enumerate(labels):
        cluster_data = features[i]
        
        low_freq_energy = cluster_data[4]
        mid_freq_energy = cluster_data[5]
        high_freq_energy = cluster_data[6]
        
        # 根据频率能量决定噪声类型
        noise_type = classify_noise_type(low_freq_energy, mid_freq_energy, high_freq_energy, low_freq_threshold, mid_freq_threshold, high_freq_threshold)
        
        # 根据原始文件夹名称和噪声类型生成新的文件夹名称
        original_folder_name = os.path.basename(original_folders[i])
        new_folder_name = f"{original_folder_name}{noise_type}"
        new_folder_path = os.path.join(output_base_folder, new_folder_name)
        
        # 创建新文件夹
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        
        # 保存图像到新的文件夹中
        original_file_path = os.path.join(original_folders[i], file_names[i])
        new_file_path = os.path.join(new_folder_path, file_names[i])
        shutil.copy2(original_file_path, new_file_path)  # 复制图片
        print(f"图片 {file_names[i]} 已保存到 {new_folder_name}")

# 聚类结果的可视化并保存图像
def visualize_and_save_clusters(features, labels, labeled_clusters, output_base_folder):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for label, cluster_label in labeled_clusters.items():
        cluster_data = reduced_data[labels == label]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"{cluster_label}", alpha=0.6)

    plt.title('聚类结果可视化')
    plt.legend()

    # 保存可视化图像
    visualization_path = os.path.join(output_base_folder, '聚类可视化结果.png')
    plt.savefig(visualization_path)
    print(f"聚类可视化图像已保存到: {visualization_path}")
    plt.show()

# 主程序入口
def process_and_save_images(image_folders, output_base_folder, fs=1000):
    images, file_names, original_folders = load_images_from_folders(image_folders)

    features = [extract_signal_features(img) for img in images]
    features = np.array(features)

    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)

    # 聚类
    labels, gmm = perform_clustering_with_bic(features)

    # 计算自适应频率阈值
    low_freq_threshold, mid_freq_threshold, high_freq_threshold = calculate_adaptive_frequency_thresholds(images[0])  # 选择一个信号进行频率阈值计算

    # 给每个聚类打标签并保存图片
    label_and_save_clusters(features, labels, file_names, original_folders, low_freq_threshold, mid_freq_threshold, high_freq_threshold, output_base_folder)

    # 为每个聚类生成可视化图像并保存
    labeled_clusters = {label: classify_noise_type(features[labels == label][0][4], 
                                                   features[labels == label][0][5], 
                                                   features[labels == label][0][6], 
                                                   low_freq_threshold, 
                                                   mid_freq_threshold, 
                                                   high_freq_threshold) for label in np.unique(labels)}
    
    visualize_and_save_clusters(features, labels, labeled_clusters, output_base_folder)


# 运行程序
if __name__ == "__main__":
    base_folder = r"D:\shudeng\波形图\聚类优化\处理后的数据\分类结果1"
    output_base_folder = r"D:\shudeng\波形图\聚类优化\处理后的数据\分类结果1\油井深度与噪音类型"
    folders = [os.path.join(base_folder, "浅井"), os.path.join(base_folder, "中深井"), os.path.join(base_folder, "深井")]
    process_and_save_images(folders, output_base_folder)
