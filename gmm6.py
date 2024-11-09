import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.signal import find_peaks, welch, stft
from scipy.stats import kurtosis, skew
import pywt
from PIL import Image  # 确保导入Image模块

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
            img = Image.open(file_path).convert('L')  # 使用 PIL.Image
            img = img.resize((64, 64))
            img_data = np.asarray(img).flatten() / 255.0
            images.append(img_data)
            file_names.append(file_name)
    return np.array(images), file_names

# 计算零交叉率（ZCR）
def zero_crossing_rate(signal):
    zcr = np.mean(np.diff(np.sign(signal)) != 0)
    return zcr

# 计算短时傅里叶变换（STFT）的平均和方差
def stft_features(signal, fs=1000, nperseg=256):
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    stft_mean = np.mean(np.abs(Zxx))
    stft_var = np.var(np.abs(Zxx))
    return stft_mean, stft_var

# 计算小波变换特征
def wavelet_features(signal, wavelet='db1'):
    coeffs = pywt.wavedec(signal, wavelet, level=4)
    wavelet_energy = np.sum([np.sum(np.square(coeff)) for coeff in coeffs])
    wavelet_var = np.var(np.hstack(coeffs))  # 计算所有小波系数的方差
    return wavelet_energy, wavelet_var

# 提取信号的主要特征，包括频率特征
def extract_signal_features(signal):
    amplitude = np.ptp(signal)  # 振幅
    std_dev = np.std(signal)    # 标准差
    energy = np.sum(signal**2)  # 能量
    peaks = len(find_peaks(signal)[0])  # 峰值计数

    skewness = skew(signal)     # 偏度
    kurt = kurtosis(signal)     # 峰度

    zcr = zero_crossing_rate(signal)    # 零交叉率
    stft_mean, stft_var = stft_features(signal)  # STFT 特征
    wavelet_energy, wavelet_var = wavelet_features(signal)  # 小波特征

    freqs, psd = welch(signal)
    low_freq_energy, mid_freq_energy, high_freq_energy = 0, 0, 0
    if len(psd) > 0:
        low_freq_energy = np.mean(psd[(freqs >= 0) & (freqs <= 200)]) if len(psd[(freqs >= 0) & (freqs <= 200)]) > 0 else 0
        mid_freq_energy = np.mean(psd[(freqs > 200) & (freqs <= 1000)]) if len(psd[(freqs > 200) & (freqs <= 1000)]) > 0 else 0
        high_freq_energy = np.mean(psd[freqs > 1000]) if len(psd[freqs > 1000]) > 0 else 0

    return [amplitude, std_dev, energy, peaks, low_freq_energy, mid_freq_energy, high_freq_energy,
            skewness, kurt, zcr, stft_mean, stft_var, wavelet_energy, wavelet_var]

# 动态调整 ZCR 阈值
def auto_adjust_zcr_threshold(features, labels):
    zcr_values = []
    for cluster_label in np.unique(labels):
        cluster_data = features[labels == cluster_label]
        zcrs = [zero_crossing_rate(data) for data in cluster_data]
        zcr_values.extend(zcrs)
    
    zcr_mean = np.mean(zcr_values)
    zcr_std = np.std(zcr_values)
    
    # 动态调整 ZCR 的阈值范围，确保有高、中、低频噪音
    zcr_threshold_high = zcr_mean + zcr_std * 0.5  # 高频阈值
    zcr_threshold_mid = zcr_mean  # 中频阈值
    zcr_threshold_low = zcr_mean - zcr_std * 0.5  # 低频阈值
    
    return zcr_threshold_high, zcr_threshold_mid, zcr_threshold_low

# 噪音分类函数，确保每个簇都有高、中、低频噪音
def classify_noise_with_dynamic_zcr(features, labels, adjusted_params, zcr_threshold_high, zcr_threshold_mid, zcr_threshold_low):
    noise_types = {}
    has_low_freq, has_mid_freq, has_high_freq = False, False, False  # 跟踪是否有高、中、低频噪音

    for cluster_label in np.unique(labels):
        cluster_data = features[labels == cluster_label]

        avg_amplitude = np.mean([np.ptp(data) for data in cluster_data])
        avg_zcr = np.mean([zero_crossing_rate(data) for data in cluster_data])

        # 振幅分类
        if avg_amplitude > adjusted_params['amplitude_threshold_high']:
            noise_strength = "强噪音"
        elif avg_amplitude > adjusted_params['amplitude_threshold_mid']:
            noise_strength = "中等噪音"
        else:
            noise_strength = "弱噪音"

        # ZCR 分类逻辑，确保分布均匀
        if avg_zcr > zcr_threshold_high:
            noise_type = "高频噪音"
            has_high_freq = True
        elif avg_zcr > zcr_threshold_mid:
            noise_type = "中频噪音"
            has_mid_freq = True
        elif avg_zcr > zcr_threshold_low:
            noise_type = "低频噪音"
            has_low_freq = True
        else:
            noise_type = "随机噪音"

        # 将噪音强度和频率类型结合
        noise_types[cluster_label] = {
            '噪音类型': f"{noise_strength} ({noise_type})"
        }

    # 确保至少有一个簇包含高、中、低频噪音
    if not has_high_freq:
        noise_types[list(noise_types.keys())[0]]['噪音类型'] = "强噪音 (高频噪音)"
    if not has_mid_freq:
        noise_types[list(noise_types.keys())[1]]['噪音类型'] = "中等噪音 (中频噪音)"
    if not has_low_freq:
        noise_types[list(noise_types.keys())[2]]['噪音类型'] = "弱噪音 (低频噪音)"
    
    return noise_types

# 使用 BIC 自动确定的 K 值进行聚类
def perform_clustering_with_bic(features, max_clusters=10):
    bic_scores = []
    best_k = 1  # 初始化最佳K值为1
    gmm_models = []
    
    # 计算1到max_clusters范围内的BIC
    for k in range(1, max_clusters + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(features)
        bic_scores.append(gmm.bic(features))
        gmm_models.append(gmm)

    # 获取 BIC 最小值对应的K值
    best_k = np.argmin(bic_scores) + 1
    best_gmm = gmm_models[np.argmin(bic_scores)]  # 获取最优的 GMM 模型

    print(f"根据 BIC，最佳聚类数 (K) 为: {best_k}")
    
    # 使用最佳 K 值进行最终的聚类
    labels = best_gmm.fit_predict(features)
    return labels, best_gmm

# 保存每个团簇中的图像，并自动命名文件夹
def save_clustered_images_by_label(image_folder, file_names, labels, noise_classification, output_folder):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        # 根据噪音分类结果命名文件夹
        noise_type = noise_classification.get(label, {}).get('噪音类型', f'cluster_{label}')
        label_folder = os.path.join(output_folder, noise_type)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)

    for file_name, label in zip(file_names, labels):
        noise_type = noise_classification.get(label, {}).get('噪音类型', f'cluster_{label}')
        dest_image_path = os.path.join(output_folder, noise_type, file_name)
        src_image_path = os.path.join(image_folder, file_name)
        img = Image.open(src_image_path)
        img.save(dest_image_path)

    print(f"所有图像已按聚类结果保存到各自的文件夹中：{output_folder}")


# 可视化聚类结果
def plot_clusters(features, labels, gmm, noise_classification, output_folder, plot_filename):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    centers = pca.transform(gmm.means_)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='簇中心')

    for i, (x, y) in enumerate(centers):
        noise_type = noise_classification.get(i, {}).get('噪音类型', '未知')
        plt.text(x, y, f'{noise_type}', fontsize=10, ha='center', color='black',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='black', boxstyle='round,pad=0.5'))

    plt.title('聚类结果')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(scatter, label='聚类标签')
    plt.legend()
    plt.savefig(os.path.join(output_folder, plot_filename))
    plt.show()

# 主程序入口
# 调用函数时应该传递所有需要的参数，包括 noise_classification
def process_images_with_combined_method(image_folder, output_folder):
    images, file_names = load_images_as_features(image_folder)

    features = [extract_signal_features(img) for img in images]
    features = np.array(features)

    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)

    # 自动选择最佳K值并进行聚类
    labels, gmm = perform_clustering_with_bic(features, max_clusters=10)

    # 动态调整 ZCR 阈值
    zcr_threshold_high, zcr_threshold_mid, zcr_threshold_low = auto_adjust_zcr_threshold(features, labels)

    # 动态调整振幅阈值
    cluster_stats = calculate_cluster_stats(features, labels)
    adjusted_params = auto_adjust_parameters(cluster_stats)

    # 噪音分类
    noise_classification = classify_noise_with_dynamic_zcr(features, labels, adjusted_params, zcr_threshold_high, zcr_threshold_mid, zcr_threshold_low)

    # 保存聚类结果中的图像
    save_clustered_images_by_label(image_folder, file_names, labels, noise_classification, output_folder)

    # 可视化聚类结果
    plot_filename = "cluster_visualization.png"
    plot_clusters(features, labels, gmm, noise_classification, output_folder, plot_filename)

    # 计算每个簇的统计信息
def calculate_cluster_stats(features, labels):
    cluster_stats = {}
    for cluster_label in np.unique(labels):
        cluster_data = features[labels == cluster_label]
        
        amplitudes = [np.ptp(data) for data in cluster_data]
        std_devs = [np.std(data) for data in cluster_data]
        energies = [np.sum(data**2) for data in cluster_data]
        
        cluster_stats[cluster_label] = {
            'avg_amplitude': np.mean(amplitudes),
            'avg_std_dev': np.mean(std_devs),
            'avg_energy': np.mean(energies),
            'max_amplitude': np.max(amplitudes),
            'min_amplitude': np.min(amplitudes)
        }
    return cluster_stats

# 自动调整参数，例如振幅阈值
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


if __name__ == "__main__":
    input_folder = r"D:\shudeng\波形图\结果1"
    output_folder = r"D:\shudeng\波形图\结果1\BIC_gmm_分类结合结果9"

    process_images_with_combined_method(input_folder, output_folder)
