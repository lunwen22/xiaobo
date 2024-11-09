import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PIL import Image

# 设置路径
input_folder = r"D:\shudeng\ProofingTool\数据\自动优化alpha信噪比123"
output_folder = r"D:\shudeng\ProofingTool\数据\聚类分析结果pca"

# 读取图像并转换为特征向量
def load_images_as_features(image_folder):
    images = []
    file_names = []
    
    for file_name in os.listdir(image_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_folder, file_name)
            
            # 读取图像
            img = Image.open(file_path)
            img = img.resize((128, 128))  # 调整大小，以便统一处理
            img_data = np.asarray(img).flatten()  # 将图像扁平化为1D向量
            
            images.append(img_data)
            file_names.append(file_name)
    
    return np.array(images), file_names

# 使用PCA进行降维
def perform_pca_on_images(images, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_images = pca.fit_transform(images)
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA explained variance ratio: {explained_variance}")
    return reduced_images

# 使用K-Means进行聚类分析
def perform_kmeans_clustering(images, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(images)
    return labels

# 可视化聚类结果
def plot_image_clustering_results(reduced_images, labels, file_names):
    plt.figure(figsize=(8, 6))
    
    plt.scatter(reduced_images[:, 0], reduced_images[:, 1], c=labels, cmap='viridis', s=100)
    plt.title('PCA 降维后的图像聚类结果')
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')
    
    # 显示每个点的文件名
    for i, file_name in enumerate(file_names):
        plt.annotate(file_name, (reduced_images[i, 0], reduced_images[i, 1]), fontsize=8)
    
    plt.colorbar(label='类别')
    plt.show()

# 主程序入口
def process_images_with_clustering(image_folder, n_clusters=3):
    # 加载图像并转换为特征
    images, file_names = load_images_as_features(image_folder)
    
    # 使用PCA进行降维
    reduced_images = perform_pca_on_images(images, n_components=2)
    
    # 执行K-Means聚类
    labels = perform_kmeans_clustering(reduced_images, n_clusters=n_clusters)
    
    # 可视化聚类结果
    plot_image_clustering_results(reduced_images, labels, file_names)

# 运行主程序
if __name__ == "__main__":
    process_images_with_clustering(input_folder, n_clusters=3)
