import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from PIL import Image
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取图像并转换为特征向量
def load_images_as_features(image_folder):
    images = []
    file_names = []
    
    for file_name in os.listdir(image_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_folder, file_name)
            
            # 读取图像并转换为灰度图像
            img = Image.open(file_path).convert('L')  # 转换为灰度
            img = img.resize((128, 128))  # 调整大小，以便统一处理
            img_data = np.asarray(img).flatten() / 255.0  # 归一化
            
            images.append(img_data)
            file_names.append(file_name)
    
    return np.array(images), file_names

# 定义自编码器模型
def build_autoencoder(input_dim, encoding_dim):
    # 输入层
    input_img = Input(shape=(input_dim,))
    
    # 编码层
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    
    # 解码层
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    # 构建自编码器模型
    autoencoder = Model(input_img, decoded)
    
    # 构建仅包含编码器的模型
    encoder = Model(input_img, encoded)
    
    # 编译模型
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    
    return autoencoder, encoder

# 使用自编码器提取低维特征
def train_autoencoder(images, encoding_dim=64, epochs=50, batch_size=256):
    input_dim = images.shape[1]
    
    # 构建自编码器
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
    
    # 训练自编码器
    autoencoder.fit(images, images, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2)
    
    # 提取编码后的低维特征
    encoded_images = encoder.predict(images)
    
    return encoded_images

# 使用K-Means进行聚类分析
def perform_kmeans_clustering(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels, kmeans

# 动态定义噪音类型，根据每个簇的统计特征
def define_noise_types(labels, reduced_images):
    noise_types = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        # 获取该簇的样本点
        cluster_points = reduced_images[labels == label]
        
        # 计算该簇的统计特征，例如标准差、均值、范围等
        std = np.std(cluster_points)
        mean = np.mean(cluster_points)
        peak_to_peak = np.ptp(cluster_points)
        
        # 自动定义噪音类型，增加更多维度判断标准（可以根据实际需要修改规则）
        if std < 0.5 and mean < 0:
            noise_types[label] = "低噪音"
        elif 0.5 <= std < 1.0 and mean > 0:
            noise_types[label] = "中等噪音"
        elif std >= 1.0:
            if peak_to_peak > 3:
                noise_types[label] = "高噪音"
            else:
                noise_types[label] = "周期性噪音"
        else:
            noise_types[label] = "未知噪音"
    
    return noise_types


# 可视化聚类结果，并标注噪音类型
def plot_image_clustering_results(reduced_images, labels, kmeans, output_folder, noise_types):
    plt.figure(figsize=(10, 8))
    
    # 用不同颜色表示不同类别，不再标注文件名
    scatter = plt.scatter(reduced_images[:, 0], reduced_images[:, 1], c=labels, cmap='viridis', s=100, alpha=0.7)
    plt.title('自编码器降维后的噪音类型聚类结果')
    plt.xlabel('最大变化方向')
    plt.ylabel('次大变化方向')
    
    # 添加颜色条表示不同噪音类型
    plt.colorbar(label='噪音类型')
    
    # 标注每个团簇的噪音类型，标注在质心位置
    for i in range(len(kmeans.cluster_centers_)):
        centroid = kmeans.cluster_centers_[i]
        plt.text(centroid[0], centroid[1], noise_types.get(i, "未知噪音"), fontsize=12, weight='bold', 
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='black', boxstyle='round,pad=0.5'))

    # 如果输出文件夹不存在，创建该文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 保存图像
    output_path = os.path.join(output_folder, 'pca_clustering_result_noise_types.png')
    plt.savefig(output_path)
    plt.close()  # 关闭图形，防止重复显示

# 主程序入口
def process_images_with_autoencoder_and_kmeans(image_folder, output_folder, n_clusters=5):
    # 加载图像并转换为特征
    images, file_names = load_images_as_features(image_folder)

    # 使用自编码器提取低维特征
    encoded_images = train_autoencoder(images, encoding_dim=64, epochs=50, batch_size=256)

    # 使用 PCA 降维（降到二维用于可视化）
    pca = PCA(n_components=2)
    reduced_images = pca.fit_transform(encoded_images)

    # 执行 K-Means 聚类
    labels, kmeans = perform_kmeans_clustering(reduced_images, n_clusters=n_clusters)

    # 自动定义噪音类型
    noise_types = define_noise_types(labels, reduced_images)

    # 可视化并保存聚类结果
    plot_image_clustering_results(reduced_images, labels, kmeans, output_folder, noise_types)

# 运行主程序
if __name__ == "__main__":
    input_folder = r"D:\shudeng\ProofingTool\数据\自动优化alpha信噪比1234 "
    output_folder = r"D:\shudeng\ProofingTool\数据\聚类分析结果pca3"
    process_images_with_autoencoder_and_kmeans(input_folder, output_folder, n_clusters=5)
