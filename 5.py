import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 定义读取图像函数
def load_image(image_path):
    image = io.imread(image_path, as_gray=True)  # 读取灰度图像
    return image

# 使用轮廓系数自动选择聚类数量
def find_optimal_clusters(image, max_clusters=10):
    flat_image = image.flatten().reshape(-1, 1)
    best_n_clusters = 2
    best_score = -1
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(flat_image)
        labels = kmeans.labels_
        score = silhouette_score(flat_image, labels)
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
    return best_n_clusters

# 聚类方法：K均值聚类
def kmeans_clustering(image, n_clusters):
    # 将图像展平为二维数组，便于聚类
    flat_image = image.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(flat_image)
    clustered_image = kmeans.labels_.reshape(image.shape)
    return clustered_image

# 保存图像
def save_image(image, output_path):
    io.imsave(output_path, image)

# 处理文件夹中的图像文件
def process_images(input_folder, output_folder, max_clusters=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            image_path = os.path.join(input_folder, file_name)
            image = load_image(image_path)
            
            # 自动选择最佳聚类数量
            best_n_clusters = find_optimal_clusters(image, max_clusters=max_clusters)
            print(f"图像 {file_name} 最佳聚类数量为: {best_n_clusters}")
            
            # 使用K均值聚类
            clustered_image = kmeans_clustering(image, n_clusters=best_n_clusters)
            
            # 保存聚类结果
            output_path = os.path.join(output_folder, f'clustered_{file_name}')
            save_image(clustered_image, output_path)

            print(f"图像 {file_name} 的聚类结果已保存至: {output_path}")

# 主程序入口
if __name__ == "__main__":
    input_folder = r"D:\shudeng\ProofingTool\数据\自动优化alpha信噪比1234"  # 输入图像的文件夹路径
    output_folder = r"D:\shudeng\ProofingTool\数据\聚类分析结果11"  # 输出聚类分析结果保存路径
    
    process_images(input_folder, output_folder, max_clusters=10)  # 最多尝试10个聚类
