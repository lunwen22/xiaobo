import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.models import ResNet18_Weights
from sklearn.metrics import silhouette_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]

# 图像预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CNN-Transformer 模型
class CNNTransformer(nn.Module):
    def __init__(self):
        super(CNNTransformer, self).__init__()
        
        # 使用预训练的 ResNet 作为 CNN 特征提取器
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # 移除全连接层
        
        # Transformer 设置
        self.transformer = nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
        )
        
        # 最后的分类器
        self.fc = nn.Linear(512, 4)  # 4 类的输出

    def forward(self, x):
        cnn_features = self.cnn(x).view(x.size(0), -1)  # 展平 CNN 输出
        transformer_output = self.transformer(cnn_features.unsqueeze(0), cnn_features.unsqueeze(0))
        output = self.fc(transformer_output.squeeze(0))
        return output

# 计算 SSE 值
def calculate_sse(features, k_range):
    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(features)
        sse.append(kmeans.inertia_)
    return sse

# 计算轮廓系数
def calculate_silhouette(features, k_range):
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        silhouette_scores.append(score)
    return silhouette_scores

# 完全自动化 K 值选择算法（移除 gap_statistic 部分）
def auto_select_k(features, max_k=10):
    k_range = range(1, max_k+1)
    
    # Step 1: 使用肘部法计算 SSE 值
    sse = calculate_sse(features, k_range)
    
    # 绘制肘部法图像
    plt.figure()
    plt.plot(k_range, sse, marker='o')
    plt.title('Elbow Method: SSE for different K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('SSE')
    plt.show()
    
    # Step 2: 通过轮廓系数进一步验证
    silhouette_scores = calculate_silhouette(features, k_range[1:])  # K 从 2 开始计算
    best_silhouette_k = k_range[1:][np.argmax(silhouette_scores)]  # 选择轮廓系数最高的K值
    
    # 绘制轮廓系数图像
    plt.figure()
    plt.plot(k_range[1:], silhouette_scores, marker='o')
    plt.title('Silhouette Scores for different K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.show()
    
    # 输出最佳K值
    elbow_k = k_range[np.argmin(np.diff(np.diff(sse)))]  # 肘部法建议的K值
    print(f"肘部法建议的K值：{elbow_k}")
    print(f"轮廓系数选择的最佳K值：{best_silhouette_k}")
    
    return best_silhouette_k  # 返回轮廓系数选择的最佳K值


# 自定义函数用于判定信号后续部分是否是低波动
def classify_low_variation_signals(features, threshold=0.01):
    low_variation_indices = []
    
    for i, signal in enumerate(features):
        tail_signal = signal[int(len(signal) * 0.20):]  # 后80%的信号
        energy = np.sum(np.square(tail_signal))
        std_dev = np.std(tail_signal)
        
        if energy < threshold or std_dev < threshold:
            low_variation_indices.append(i)
    
    return low_variation_indices

# 保存聚类结果到 CSV 文件
def save_cluster_results_to_csv(image_paths, labels, output_csv):
    df = pd.DataFrame({
        'image_path': image_paths,
        'cluster': labels
    })
    df.to_csv(output_csv, index=False)
    print(f"聚类结果已保存到 {output_csv}")

# 使用 PCA 或 t-SNE 进行聚类可视化
def visualize_clusters_pca_tsne(features, labels, method='pca'):
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced_features = reducer.fit_transform(features)
        title = 'PCA-based Clustering Visualization'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, n_iter=300)
        reduced_features = reducer.fit_transform(features)
        title = 't-SNE-based Clustering Visualization'
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=50)
    plt.colorbar(scatter)
    plt.title(title, fontsize=14)
    plt.xlabel("Component 1", fontsize=12)
    plt.ylabel("Component 2", fontsize=12)
    plt.show()

# 可视化并进行傅里叶变换分析
def visualize_clusters(image_paths, labels, output_dir, num_clusters):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cluster_labels = {i: f"Cluster {i} Noise" for i in range(num_clusters)}
    low_variation_label = "Low-Variation Signal"

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        label = labels[i]
        if label == 'Low-Variation Signal':
            label_name = low_variation_label
        else:
            label_name = cluster_labels[label]
        
        output_path = os.path.join(output_dir, f"cluster_{label}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cv2.putText(img, f"{label_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(output_path, os.path.basename(img_path)), img)

        waveform = img.mean(axis=0)
        plt.figure()
        plt.title(f"{label_name}")
        plt.plot(np.arange(len(waveform)), waveform)
        plt.savefig(os.path.join(output_path, f"waveform_{label}.png"))
        plt.close()

        fft_result = np.fft.fft(waveform)
        freqs = np.fft.fftfreq(len(fft_result))
        magnitude = np.abs(fft_result)

        plt.figure()
        plt.title(f"{label_name} FFT Spectrum")
        plt.plot(freqs, magnitude)
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.savefig(os.path.join(output_path, f"fft_spectrum_{label}.png"))
        plt.close()
# 主函数
def main():
    # 定义图片文件夹路径和输出文件夹路径
    image_dir = r'D:\shudeng\boxongtu\jieguoyi'  # 替换为您的图片文件夹路径
    output_dir = r'D:\shudeng\boxongtu\bufenleidejulei'  # 替换为保存聚类结果的文件夹路径
    output_csv = r'D:\shudeng\boxongtu\bufenleidejulei\result.csv'  # CSV文件保存路径

    # 加载数据集
    dataset = ImageDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 初始化 CNN-Transformer 模型
    model = CNNTransformer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    features = []
    image_paths = []

    # 提取特征
    with torch.no_grad():
        for images, img_names in dataloader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            image_paths.extend([os.path.join(image_dir, img_name) for img_name in img_names])

    features = np.vstack(features)

    # 标准化特征
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 自动选择最佳 K 值
    final_k_value = auto_select_k(features)

    # 使用KMeans进行最终聚类
    kmeans = KMeans(n_clusters=final_k_value, random_state=0)
    labels = kmeans.fit_predict(features)

    # 检查哪些信号属于“低波动信号”
    low_variation_indices = classify_low_variation_signals(features, threshold=0.01)
    for idx in low_variation_indices:
        labels[idx] = 'Low-Variation Signal'  # 将低波动信号归类为新的类别

    # 保存聚类结果到CSV
    save_cluster_results_to_csv(image_paths, labels, output_csv)

    # 使用 PCA 或 t-SNE 可视化聚类
    visualize_clusters_pca_tsne(features, labels, method='pca')  # 使用 PCA 可视化
    visualize_clusters_pca_tsne(features, labels, method='tsne')  # 使用 t-SNE 可视化

    # 可视化并进行傅里叶变换分析
    visualize_clusters(image_paths, labels, output_dir, final_k_value)

if __name__ == "__main__":
    main()
