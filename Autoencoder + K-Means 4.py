import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from PIL import Image
import matplotlib
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取图像并转换为特征向量
def load_images_as_features(image_folder):
    images = []
    file_names = []
    
    for file_name in os.listdir(image_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_folder, file_name)
            
            # 读取图像并转换为灰度图像
            img = Image.open(file_path).convert('L')
            img = img.resize((128, 128))
            img_data = np.asarray(img).flatten() / 255.0
            
            images.append(img_data)
            file_names.append(file_name)
    
    return np.array(images), file_names

# 定义自编码器模型
def build_autoencoder(input_dim, encoding_dim, layers=2, activation='leaky_relu'):
    input_img = Input(shape=(input_dim,))
    
    encoded = input_img
    for _ in range(layers):
        if activation == 'leaky_relu':
            encoded = Dense(encoding_dim)(encoded)
            encoded = LeakyReLU(alpha=0.1)(encoded)
        elif activation == 'tanh':
            encoded = Dense(encoding_dim, activation='tanh')(encoded)
        else:
            encoded = Dense(encoding_dim, activation='relu')(encoded)
    
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    
    return autoencoder, encoder

# 使用自编码器提取低维特征
def train_autoencoder(images, encoding_dim=64, epochs=50, batch_size=256, layers=2, activation='leaky_relu'):
    input_dim = images.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim, layers, activation)
    
    autoencoder.fit(images, images, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2)
    
    encoded_images = encoder.predict(images)
    return encoded_images

# 使用K-Means进行聚类分析
def perform_kmeans_clustering(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels, kmeans

# 动态阈值函数
def adaptive_threshold(signal):
    return np.mean(signal) + 2 * np.std(signal)

# 定义噪音类型，基于频率、能量、周期性和非平稳性
def define_noise_types(labels, reduced_images):
    noise_types = {}
    for label in np.unique(labels):
        cluster_points = reduced_images[labels == label]

        # 确保 cluster_points 是 1-D 数组
        cluster_points = np.asarray(cluster_points).flatten()

        # 计算统计特征：标准差、均值、峰值范围、偏度、峰度等
        std = np.std(cluster_points)
        mean = np.mean(cluster_points)
        peak_to_peak = np.ptp(cluster_points)
        skewness = skew(cluster_points)
        kurt = kurtosis(cluster_points)

        # 计算频率和能量特征，自动调整 nperseg
        signal_length = len(cluster_points)
        nperseg = min(256, signal_length // 2) if signal_length > 2 else 1

        if signal_length > 2:
            freqs, psd = welch(cluster_points, nperseg=nperseg)
            if len(psd) > 1 and len(freqs) > np.argmax(psd):
                dominant_freq = freqs[np.argmax(psd)]
                total_energy = np.sum(psd)
            else:
                dominant_freq = 0
                total_energy = 0
        else:
            dominant_freq = 0
            total_energy = 0

        # 检查周期性
        peaks, _ = find_peaks(cluster_points)
        if signal_length > 0:
            periodicity = len(peaks) / signal_length
        else:
            periodicity = 0  # 如果信号长度为0，设置周期性为0

        # 检查非平稳性（使用差分标准差的变化来检测）
        if signal_length > 1:
            non_stationarity = np.var(np.diff(cluster_points))  # 使用差分的方差作为非平稳性的量度
        else:
            non_stationarity = 0  # 如果信号过短，设置非平稳性为0

        # 动态调整阈值
        threshold = adaptive_threshold(cluster_points)

        # 根据频率、能量、周期性、非平稳性划分噪音类型
        if total_energy < threshold and dominant_freq < 50:
            noise_types[label] = "低频低能量噪音"
        elif total_energy >= threshold and dominant_freq < 50:
            noise_types[label] = "低频高能量噪音"
        elif 50 <= dominant_freq < 200 and total_energy >= threshold:
            noise_types[label] = "中频中等能量噪音"
        elif dominant_freq >= 200 and total_energy >= threshold and periodicity > 0.1:
            noise_types[label] = "高频高能量周期性噪音"
        elif dominant_freq >= 200 and total_energy < threshold and non_stationarity > 0.05:
            noise_types[label] = "高频低能量非平稳噪音"
        else:
            noise_types[label] = "随机噪音"

    return noise_types

# 可视化聚类结果，并标注噪音类型
def plot_image_clustering_results(reduced_images, labels, kmeans, output_folder, noise_types):
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(reduced_images[:, 0], reduced_images[:, 1], c=labels, cmap='viridis', s=100, alpha=0.7)
    plt.title('自编码器降维后的噪音类型聚类结果')
    plt.xlabel('最大变化方向')
    plt.ylabel('次大变化方向')
    
    plt.colorbar(label='噪音类型')
    
    for i in range(len(kmeans.cluster_centers_)):
        centroid = kmeans.cluster_centers_[i]
        plt.text(centroid[0], centroid[1], noise_types.get(i, "未知噪音"), fontsize=12, weight='bold', 
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='black', boxstyle='round,pad=0.5'))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_path = os.path.join(output_folder, 'pca_clustering_result_noise_types.png')
    plt.savefig(output_path)
    plt.close()

# 主程序入口
def process_images_with_autoencoder_and_kmeans(image_folder, output_folder, n_clusters=5, layers=2, activation='leaky_relu'):
    images, file_names = load_images_as_features(image_folder)
    encoded_images = train_autoencoder(images, encoding_dim=64, epochs=50, batch_size=256, layers=layers, activation=activation)
    
    pca = PCA(n_components=2)
    reduced_images = pca.fit_transform(encoded_images)

    labels, kmeans = perform_kmeans_clustering(reduced_images, n_clusters=n_clusters)
    noise_types = define_noise_types(labels, reduced_images)
    
    plot_image_clustering_results(reduced_images, labels, kmeans, output_folder, noise_types)

if __name__ == "__main__":
    input_folder = r"D:\shudeng\ProofingTool\数据\自动优化alpha信噪比1234"
    output_folder = r"D:\shudeng\ProofingTool\数据\聚类分析结果pca4"
    process_images_with_autoencoder_and_kmeans(input_folder, output_folder, n_clusters=6, layers=3, activation='leaky_relu')
