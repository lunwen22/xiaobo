import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.signal import welch, stft, find_peaks
from scipy.stats import kurtosis, skew
import pywt
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
            img = img.resize((64, 64))  # 假设图像统一大小为 64x64
            img_data = np.asarray(img).flatten() / 255.0
            images.append(img_data)
            file_names.append(file_name)
    return np.array(images), file_names

# 动态设定小波分解层数
def get_wavelet_decomposition_level(signal_length):
    max_level = int(np.log2(signal_length))  # 最大分解层数
    return min(max_level, 5)  # 限制最大分解层数为5

# 绘制信号的频谱图
def plot_signal_spectrum(signal, fs=1000):
    freqs, psd = welch(signal, fs=fs, nperseg=1024)  # 使用更高的 nperseg 捕捉更多频率
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, psd)
    plt.title('信号的功率谱密度')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度')
    plt.grid(True)
    plt.show()

# 提取信号特征时捕捉更细致的高频信号
def extract_signal_features(signal, fs=1000):
    amplitude = np.ptp(signal)  # 振幅
    std_dev = np.std(signal)    # 标准差
    energy = np.sum(signal**2)  # 能量
    peaks = len(find_peaks(signal)[0])  # 峰值计数

    skewness = skew(signal)     # 偏度
    kurt = kurtosis(signal)     # 峰度

    # 使用更精细的频谱分析参数
    freqs, psd = welch(signal, fs=fs, nperseg=1024)  # 捕捉更细致的频谱
    low_freq_energy = np.mean(psd[(freqs >= 0) & (freqs <= 100)])  # 低频：0-100Hz
    mid_freq_energy = np.mean(psd[(freqs > 100) & (freqs <= 300)])  # 中频：100-300Hz
    high_freq_energy = np.mean(psd[freqs > 300])  # 高频：300Hz 以上

    # STFT 特征
    f, t, Zxx = stft(signal, fs=fs)
    stft_mean = np.mean(np.abs(Zxx))
    stft_var = np.var(np.abs(Zxx))

    # 小波变换特征
    signal_length = len(signal)
    wavelet_level = get_wavelet_decomposition_level(signal_length)
    coeffs = pywt.wavedec(signal, 'coif5', level=wavelet_level)
    wavelet_energy = np.sum([np.sum(np.square(coeff)) for coeff in coeffs])
    wavelet_var = np.var(np.hstack(coeffs))

    return [amplitude, std_dev, energy, peaks, low_freq_energy, mid_freq_energy, high_freq_energy,
            stft_mean, stft_var, wavelet_energy, wavelet_var, skewness, kurt]

# 动态调整阈值分类
def classify_noise(features, labels, low_freq_threshold, mid_freq_threshold, high_freq_threshold):
    noise_types = {}

    for cluster_label in np.unique(labels):
        cluster_data = features[labels == cluster_label]
        
        avg_low_freq_energy = np.mean([data[4] for data in cluster_data])
        avg_mid_freq_energy = np.mean([data[5] for data in cluster_data])
        avg_high_freq_energy = np.mean([data[6] for data in cluster_data])
        avg_amplitude = np.mean([data[0] for data in cluster_data])

        # 噪音强度分类
        noise_strength = "强噪音" if avg_amplitude > np.percentile([data[0] for data in features], 75) else "弱噪音"

        # 动态调整的噪音频率分类：调整阈值以确保高频出现
        if avg_high_freq_energy > high_freq_threshold:
            noise_type = "高频噪音"
        elif avg_mid_freq_energy > mid_freq_threshold:
            noise_type = "中频噪音"
        else:
            noise_type = "低频噪音"

        noise_types[cluster_label] = {'噪音类型': f"{noise_strength} ({noise_type})"}

    return noise_types

# 计算频率阈值，确保每个频段都有噪音分类
def calculate_frequency_thresholds(features):
    # 获取所有低频、中频和高频能量的分布
    low_freq_energies = [data[4] for data in features]
    mid_freq_energies = [data[5] for data in features]
    high_freq_energies = [data[6] for data in features]

    # 设置阈值，确保每个频率段的噪声被捕捉到
    low_freq_threshold = np.percentile(low_freq_energies, 60)  # 提高低频噪声的敏感度
    mid_freq_threshold = np.percentile(mid_freq_energies, 60)  # 提高中频噪声的敏感度
    high_freq_threshold = np.percentile(high_freq_energies, 40)  # 降低高频噪声的敏感度

    # 返回调整后的阈值
    return low_freq_threshold, mid_freq_threshold, high_freq_threshold

# 保存分类结果的图片
def save_clustered_images(image_folder, file_names, labels, noise_types, output_folder):
    for file_name, label in zip(file_names, labels):
        noise_type = noise_types[label]['噪音类型']
        dest_folder = os.path.join(output_folder, noise_type)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        
        src_image_path = os.path.join(image_folder, file_name)
        dest_image_path = os.path.join(dest_folder, file_name)
        img = Image.open(src_image_path)
        img.save(dest_image_path)

    print("图片保存完成，按噪音类型分类存放。")

# 二次聚类并打标签
def secondary_clustering(features, labels, noise_types, output_folder):
    clustered_features = {}
    for label in np.unique(labels):
        clustered_features[label] = features[labels == label]

    for label, cluster_data in clustered_features.items():
        kmeans = KMeans(n_clusters=3, random_state=42)  # 二次聚类
        secondary_labels = kmeans.fit_predict(cluster_data)

        # 计算二次聚类后的阈值
        low_freq_threshold = np.percentile([data[4] for data in cluster_data], 75)
        mid_freq_threshold = np.percentile([data[5] for data in cluster_data], 75)
        high_freq_threshold = np.percentile([data[6] for data in cluster_data], 75)

        noise_types_secondary = classify_noise(cluster_data, secondary_labels, low_freq_threshold, mid_freq_threshold, high_freq_threshold)
        
        # 保存二次分类结果并打标签
        for sec_label, noise_info in noise_types_secondary.items():
            noise_type_folder = os.path.join(output_folder, noise_info['噪音类型'])
            if not os.path.exists(noise_type_folder):
                os.makedirs(noise_type_folder)

# BIC 自动确定最佳 K 值进行聚类
def perform_clustering_with_bic(features, max_clusters=10):
    bic_scores = []
    gmm_models = []
    
    for k in range(1, max_clusters + 10):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(features)
        bic_scores.append(gmm.bic(features))
        gmm_models.append(gmm)

    best_k = np.argmin(bic_scores) + 1
    best_gmm = gmm_models[np.argmin(bic_scores)]
    print(f"根据 BIC，最佳聚类数 (K) 为: {best_k}")
    
    labels = best_gmm.fit_predict(features)
    return labels, best_gmm

# 主程序入口
def process_images_with_combined_method(image_folder, output_folder):
    images, file_names = load_images_as_features(image_folder)

    features = [extract_signal_features(img) for img in images]
    features = np.array(features)

    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)

    # 绘制频谱图检查
    plot_signal_spectrum(images[0])  # 以第一个图像对应的信号为例检查频谱

    # 第一次聚类
    labels, gmm = perform_clustering_with_bic(features, max_clusters=10)

    # 计算频率能量的阈值，使用新定义的逻辑
    low_freq_threshold, mid_freq_threshold, high_freq_threshold = calculate_frequency_thresholds(features)

    # 分类并打标签
    noise_types = classify_noise(features, labels, low_freq_threshold, mid_freq_threshold, high_freq_threshold)

    # 二次聚类
    secondary_clustering(features, labels, noise_types, output_folder)

    # 保存分类结果
    save_clustered_images(image_folder, file_names, labels, noise_types, output_folder)

if __name__ == "__main__":
    input_folder = r"D:\shudeng\波形图\结果1\BIC_gmm_分类结合结果9\zaosheng"
    output_folder = r"D:\shudeng\波形图\结果1\BIC_gmm_分类结合结果9\二次聚类结果1"
    process_images_with_combined_method(input_folder, output_folder)
