import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from skimage.feature import graycomatrix, graycoprops  # GLCM 特征
from PIL import Image  # Pillow，用于图像处理
import tensorflow as tf  # 用于深度学习特征提取
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# 设置中文字体支持
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取图像并转换为 GLCM 纹理特征
def load_images_as_features(image_folder):
    images = []
    file_names = []
    
    for file_name in os.listdir(image_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_folder, file_name)

            if not os.path.exists(file_path):
                print(f"文件 {file_path} 不存在，请检查路径。")
                continue

            try:
                # 使用 Pillow 读取图像，转为灰度图像
                img = Image.open(file_path).convert('L')  # 转为灰度图像
                img = img.resize((128, 128))  # 调整图像大小
                img_data = np.asarray(img)  # 将图像转换为 numpy 数组

                # 使用 GLCM 提取纹理特征
                glcm = graycomatrix(img_data, [1], [0], symmetric=True, normed=True)
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                correlation = graycoprops(glcm, 'correlation')[0, 0]
                
                img_features = [contrast, homogeneity, energy, correlation]
                images.append(img_features)
                file_names.append(file_name)
            except Exception as e:
                print(f"读取图像文件时出错: {file_path}，错误信息: {e}")
                continue

    return np.array(images), file_names

# 使用 VGG16 提取深度学习特征
def extract_deep_features(image_folder):
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    deep_features = []
    file_names = []
    
    for file_name in os.listdir(image_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_folder, file_name)
            
            if not os.path.exists(file_path):
                print(f"文件 {file_path} 不存在，请检查路径。")
                continue

            try:
                # 使用 Pillow 读取图像并转换为 RGB 格式
                img = Image.open(file_path).convert('RGB')
                img = img.resize((128, 128))  # 调整图像大小
                img_array = np.asarray(img)  # 将图像转换为 numpy 数组
                
                # 预处理图像以符合 VGG16 模型的输入
                img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))
                
                # 提取深度特征
                features = vgg_model.predict(img_preprocessed)
                deep_features.append(features.flatten())  # 展平成一维向量
                file_names.append(file_name)
            except Exception as e:
                print(f"读取图像文件时出错: {file_path}，错误信息: {e}")
                continue

    return np.array(deep_features), file_names

# 自动选择最佳 K 值（基于 Davies-Bouldin 指数）
def select_best_k_by_pfa(features, max_clusters=10):
    db_scores = []
    k_range = range(2, max_clusters + 1)  # K=1时无意义

    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42, init_params='kmeans', n_init=10)
        labels = gmm.fit_predict(features)

        # 计算 Davies-Bouldin 指数
        db_index = davies_bouldin_score(features, labels)
        db_scores.append(db_index)

    # 绘制 Davies-Bouldin 指数随 K 值的变化曲线
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, db_scores, label='Davies-Bouldin Index', marker='o')
    plt.xlabel('聚类数量 (K)')
    plt.ylabel('Davies-Bouldin 指数')
    plt.title('Davies-Bouldin 指数 随 K 值的变化')
    plt.legend()
    plt.show()

    # 自动选择最小 Davies-Bouldin 指数对应的 K 值
    best_k = k_range[np.argmin(db_scores)]
    print(f"根据Davies-Bouldin指数选择的最佳K值为: {best_k}")
    return best_k

# 使用 PCA 降维并聚类
def subspace_gmm_clustering(features, n_clusters):
    pca = PCA(n_components=2)  # 使用 PCA 降维至2D
    reduced_features = pca.fit_transform(features)
    
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
    labels = gmm.fit_predict(reduced_features)
    
    return labels, reduced_features

# 可视化聚类结果并标注中文
def plot_and_save_results(reduced_features, labels, output_folder):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=100, alpha=0.7)
    plt.title('降维后的噪音类型聚类结果')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(label='簇')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, 'gmm_clustering_result.png')
    plt.savefig(output_path)
    plt.show()

# 主程序
def process_images(image_folder, output_folder, max_clusters=10):
    # 加载图像并提取特征 (GLCM + VGG16)
    images_glcm, file_names_glcm = load_images_as_features(image_folder)
    images_deep, file_names_deep = extract_deep_features(image_folder)

    # 确保 GLCM 和 VGG16 特征的样本数相同
    if images_glcm.shape[0] != images_deep.shape[0]:
        raise ValueError("GLCM 特征和 VGG16 特征的样本数量不一致。")

    # 合并 GLCM 特征和 VGG16 深度学习特征
    combined_features = np.hstack([images_glcm, images_deep])

    # 特征标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(combined_features)

    # 自动选择最佳 K 值
    best_k = select_best_k_by_pfa(features_scaled, max_clusters=max_clusters)

    # 使用 PCA 降维并进行 GMM 聚类
    labels, reduced_features = subspace_gmm_clustering(features_scaled, best_k)

    # 保存并可视化聚类结果
    plot_and_save_results(reduced_features, labels, output_folder)

if __name__ == "__main__":
    input_folder = r"D:\shudeng\波形图\结果1"
    output_folder = r"D:\shudeng\波形图\结果1\GMM_聚类结果"
    
    process_images(input_folder, output_folder)
