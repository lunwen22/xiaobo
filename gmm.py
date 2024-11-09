import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from PIL import Image
from scipy.stats import skew, kurtosis
from scipy.signal import welch

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
            img = img.resize((128, 128))
            img_data = np.asarray(img).flatten() / 255.0
            images.append(img_data)
            file_names.append(file_name)
    return np.array(images), file_names

# 计算零交叉率
def zero_crossing_rate(signal):
    return np.sum(np.diff(np.sign(signal)) != 0)

# 计算信号的振幅差分
def amplitude_difference(signal):
    return np.sum(np.abs(np.diff(signal)))

# 提取信号的频域特征
def extract_frequency_features(signal):
    signal_length = len(signal)
    nperseg = min(256, signal_length // 2) if signal_length > 2 else 1
    if signal_length > 2:
        freqs, psd = welch(signal, nperseg=nperseg)
        peak_freq = freqs[np.argmax(psd)]  # 频谱峰值
        total_energy = np.sum(psd)  # 总能量
        band_energy = np.sum(psd[freqs < 0.1])  # 低频能量（举例）
    else:
        peak_freq = 0
        total_energy = 0
        band_energy = 0
    return peak_freq, total_energy, band_energy

# 提取每个信号的统计特征和频域特征
def extract_features(signal):
    std = np.std(signal)
    mean = np.mean(signal)
    peak_to_peak = np.ptp(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    zcr = zero_crossing_rate(signal)
    amplitude_diff = amplitude_difference(signal)

    # 提取频率相关特征
    peak_freq, total_energy, band_energy = extract_frequency_features(signal)

    return [std, mean, peak_to_peak, skewness, kurt, zcr, amplitude_diff, total_energy, peak_freq, band_energy]

# 使用手肘法找到最佳K值
def elbow_method(features, max_clusters=10):
    sse = []
    for k in range(1, max_clusters + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(features)
        sse.append(gmm.score(features))  # 使用高斯混合模型的得分代替SSE
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), sse, marker='o')
    plt.title('手肘法确定最佳聚类数量')
    plt.xlabel('聚类数量')
    plt.ylabel('模型得分')
    plt.show()

    best_k = int(input("根据手肘法图形选择最佳聚类数量: "))
    return best_k

# 使用GMM进行聚类
def perform_gmm_clustering(features, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = gmm.fit_predict(features)
    return labels, gmm

# 使用随机森林分类器并进行超参数优化
def train_model_with_grid_search(X_train, y_train, model_type='rf'):
    if model_type == 'rf':
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [10, 30, 50, None]
        }
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'svm':
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
        model = SVC(random_state=42)
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print(f"最佳参数: {grid_search.best_params_}")
    return grid_search.best_estimator_

# 定义噪声类别
noise_labels = {
    0: "低频噪声",
    1: "高频噪声",
    2: "中频噪声",
    3: "复杂噪声",
    4: "随机噪声"
}

# 保存每个聚类中的图像到对应的文件夹
def save_clustered_images_by_label(image_folder, file_names, labels, output_folder):
    # 为每个聚类创建文件夹
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_folder = os.path.join(output_folder, f'cluster_{label}')
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)

    # 逐个图像保存到对应的聚类文件夹
    for file_name, label in zip(file_names, labels):
        src_image_path = os.path.join(image_folder, file_name)
        dest_image_path = os.path.join(output_folder, f'cluster_{label}', file_name)
        img = Image.open(src_image_path)
        img.save(dest_image_path)

    print(f"所有图像已按聚类结果保存到各自的文件夹中：{output_folder}")

# 可视化聚类结果
def plot_image_clustering_results(reduced_images, labels, output_folder, noise_labels):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_images[:, 0], reduced_images[:, 1], c=labels, cmap='viridis', s=100, alpha=0.7)
    plt.title('降维后的噪音类型聚类结果')
    plt.xlabel('最大变化方向')
    plt.ylabel('次大变化方向')
    plt.colorbar(label='噪音类型')

    for i in range(len(np.unique(labels))):
        plt.text(reduced_images[labels == i, 0].mean(),
                 reduced_images[labels == i, 1].mean(),
                 noise_labels.get(i, "未知噪音"),
                 fontsize=12, weight='bold', 
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='black', boxstyle='round,pad=0.5'))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, 'pca_clustering_result_noise_types.png')
    plt.savefig(output_path)
    plt.close()

# 主程序入口：结合手肘法、GMM和分类器进行分类
def process_images_with_combined_method(image_folder, output_folder):
    images, file_names = load_images_as_features(image_folder)
    features = [extract_features(img) for img in images]
    
    # 使用手肘法确定最佳K值
    best_k = elbow_method(features, max_clusters=10)
    
    # 使用GMM进行聚类
    labels, gmm = perform_gmm_clustering(features, n_clusters=best_k)

    # 保存每个聚类中的图像到对应的文件夹
    save_clustered_images_by_label(image_folder, file_names, labels, output_folder)

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # 使用SMOTE对训练数据进行过采样
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 选择分类器（随机森林或SVM）
    clf = train_model_with_grid_search(X_train_resampled, y_train_resampled, model_type='rf')

    # 预测并输出结果
    y_pred = clf.predict(X_test)
    print(f"模型的准确率: {accuracy_score(y_test, y_pred)}")
    print(f"分类报告:\n{classification_report(y_test, y_pred, zero_division=1)}")

    # 使用PCA降维并可视化聚类结果
    pca = PCA(n_components=2)
    reduced_images = pca.fit_transform(features)
    plot_image_clustering_results(reduced_images, labels, output_folder, noise_labels)

if __name__ == "__main__":
    input_folder = r"D:\shudeng\波形图\结果1"
    output_folder = r"D:\shudeng\波形图\结果1\手肘法_gmm_分类结合结果"
    
    process_images_with_combined_method(input_folder, output_folder)
