import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from scipy.signal import welch, find_peaks
from scipy.stats import kurtosis, skew
from PIL import Image

# 设置中文字体支持
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 假设声波速度 v 和采样频率 fs
v_sound = 1500  # 声波在油井中的传播速度 (m/s)
fs = 1000       # 采样频率 (Hz)

# 液面值转换函数
def liquid_change(liquid, sonic):
    real_liquid = liquid / 470 / 2 * sonic
    return real_liquid

# 井深度分类
def classify_well_depth(depth):
    if depth <= 2000:
        return "浅井"
    elif depth <= 4500:
        return "中深井"
    elif depth <= 6000:
        return "深井"
    else:
        return "超深井"

# 噪声类型分类
def classify_noise_type(low_freq_energy, mid_freq_energy, high_freq_energy, low_freq_threshold, mid_freq_threshold, high_freq_threshold):
    if high_freq_energy > high_freq_threshold:
        return "高频噪声"
    elif mid_freq_energy > mid_freq_threshold:
        return "中频噪声"
    else:
        return "低频噪声"

# 液面信号的转换（检测液面点）
def detect_well_depth_by_peaks(signal, fs=1000, v_sound=1500):
    valleys, _ = find_peaks(-signal, height=None, distance=200)  # 找到凹面点
    if len(valleys) > 0:
        first_valley = valleys[0]
        time_delay = first_valley / fs
        depth = (v_sound * time_delay) / 2  # 往返时间除以2
        well_type = classify_well_depth(depth)
        return depth, well_type, first_valley
    else:
        return None, "未知", None

# 读取图像并转换为特征向量
def load_images_as_features(image_folder):
    images = []
    file_names = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_folder, file_name)
            img = Image.open(file_path).convert('L')
            img = img.resize((64, 64))  # 假设图像统一大小为 64x64
            img_data = np.asarray(img).flatten() / 255.0
            images.append(img_data)
            file_names.append(file_name)
    return np.array(images), file_names

# 提取信号特征
def extract_signal_features(signal, fs=1000):
    amplitude = np.ptp(signal)  # 振幅
    std_dev = np.std(signal)    # 标准差
    energy = np.sum(signal**2)  # 能量
    peaks = len(find_peaks(signal)[0])  # 峰值计数

    skewness = skew(signal)     # 偏度
    kurt = kurtosis(signal)     # 峰度

    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    low_freq_energy = np.mean(psd[(freqs >= 0) & (freqs <= 100)])  # 低频：0-100Hz
    mid_freq_energy = np.mean(psd[(freqs > 100) & (freqs <= 300)])  # 中频：100-300Hz
    high_freq_energy = np.mean(psd[freqs > 300])  # 高频：300Hz 以上

    return [amplitude, std_dev, energy, peaks, low_freq_energy, mid_freq_energy, high_freq_energy]

# 自动创建不存在的文件夹
def create_directory_if_not_exists(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
    except OSError as error:
        print(f"创建文件夹 {folder_path} 失败，错误信息: {error}")

# 计算频率阈值，确保每个频段都有噪音分类
def calculate_frequency_thresholds(signal, fs=1000):
    # 使用 Welch 方法计算信号的频率和功率谱密度
    freqs, psd = welch(signal, fs=fs, nperseg=1024)

    low_freq_energies = psd[(freqs >= 0) & (freqs <= 100)]
    mid_freq_energies = psd[(freqs > 100) & (freqs <= 300)]
    high_freq_energies = psd[freqs > 300]

    # 计算各频段的平均能量阈值
    low_freq_threshold = np.mean(low_freq_energies)
    mid_freq_threshold = np.mean(mid_freq_energies)
    high_freq_threshold = np.mean(high_freq_energies)

    return low_freq_threshold, mid_freq_threshold, high_freq_threshold

# BIC 自动确定最佳 K 值进行聚类
def perform_clustering_with_bic(features, max_clusters=10):
    bic_scores = []
    gmm_models = []
    
    # 尝试从 1 到 max_clusters 的聚类数进行训练
    for k in range(1, max_clusters + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(features)
        bic_scores.append(gmm.bic(features))
        gmm_models.append(gmm)

    # 找到 BIC 值最小的位置，确定最佳聚类数
    best_k = np.argmin(bic_scores) + 1
    best_gmm = gmm_models[np.argmin(bic_scores)]
    print(f"根据 BIC，最佳聚类数 (K) 为: {best_k}")
    
    # 使用最佳 GMM 进行聚类
    labels = best_gmm.fit_predict(features)
    return labels, best_gmm

# 给每个类别打标签
def label_clusters(features, labels, depth_list, low_freq_threshold, mid_freq_threshold, high_freq_threshold):
    labeled_clusters = {}
    
    for label in np.unique(labels):
        cluster_data = features[labels == label]
        cluster_depth = depth_list[labels == label]
        
        avg_low_freq_energy = np.mean([data[4] for data in cluster_data])
        avg_mid_freq_energy = np.mean([data[5] for data in cluster_data])
        avg_high_freq_energy = np.mean([data[6] for data in cluster_data])
        
        # 根据噪声频率类型
        noise_type = classify_noise_type(avg_low_freq_energy, avg_mid_freq_energy, avg_high_freq_energy, low_freq_threshold, mid_freq_threshold, high_freq_threshold)

        # 根据井深分类
        avg_depth = np.mean(cluster_depth)
        well_type = classify_well_depth(avg_depth)
        
        labeled_clusters[label] = f"{noise_type}{well_type}"
    
    return labeled_clusters

# 保存分类结果的图片
def save_clustered_images(image_folder, file_names, labels, labeled_clusters, output_folder):
    for file_name, label in zip(file_names, labels):
        noise_type = labeled_clusters[label]
        dest_folder = os.path.join(output_folder, noise_type)
        
        # 使用函数确保文件夹存在
        create_directory_if_not_exists(dest_folder)

        src_image_path = os.path.join(image_folder, file_name)
        dest_image_path = os.path.join(dest_folder, file_name)
        
        img = Image.open(src_image_path)
        img.save(dest_image_path)

    print("图片保存完成，按噪音类型分类存放。")

# 主程序入口
def process_images_with_combined_method(image_folder, output_folder, sonic=1500):
    images, file_names = load_images_as_features(image_folder)

    features = [extract_signal_features(img) for img in images]
    features = np.array(features)

    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)

    depth_list = []
    for img in images:
        signal = img  # 假设图像已转换为信号

        # 检测井深和液面点
        depth, well_type, _ = detect_well_depth_by_peaks(signal)
        if depth is not None:
            depth = liquid_change(depth, sonic)
            depth_list.append(depth)
        else:
            depth_list.append(0)  # 无法检测到深度时的占位符

    # 聚类
    labels, gmm = perform_clustering_with_bic(features)

    # 计算频率阈值
    low_freq_threshold, mid_freq_threshold, high_freq_threshold = calculate_frequency_thresholds(images[0])  # 选择一个信号进行频率阈值计算

    # 给每个聚类打标签
    labeled_clusters = label_clusters(features, labels, np.array(depth_list), low_freq_threshold, mid_freq_threshold, high_freq_threshold)

    # 保存按标签分类的图像
    save_clustered_images(image_folder, file_names, labels, labeled_clusters, output_folder)

# 运行程序
if __name__ == "__main__":
    input_folder = r"D:\shudeng\ProofingTool\聚类图象\自动优化alpha信噪比123"
    output_folder = r"D:\shudeng\ProofingTool\聚类图象\自动优化alpha信噪比123\分类结果"
    process_images_with_combined_method(input_folder, output_folder)
