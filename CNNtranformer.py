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
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler  # 为确保特征标准化
from torchvision.models import ResNet18_Weights

# 自定义数据集类，用于加载文件夹中的图片
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_name)  # 使用 OpenCV 加载图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]

# 图像预处理步骤，包括调整尺寸、数据增强、转换为张量和归一化
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(15),      # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 使用预训练的 ResNet 作为特征提取器
class PretrainedResNetExtractor(nn.Module):
    def __init__(self):
        super(PretrainedResNetExtractor, self).__init__()
        # 加载预训练的 ResNet 模型，并使用新的 weights 参数
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # 移除 ResNet 的最后一层全连接层，仅保留特征提取部分
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)  # 展平为二维张量

# 使用手肘法确定最佳K值
def determine_best_k(features, max_k=10):
    sse = []
    k_values = list(range(1, max_k + 1))

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(features)
        sse.append(kmeans.inertia_)  # inertia_ 是 KMeans 内部的 SSE 计算

    # 绘制手肘法曲线
    plt.plot(k_values, sse, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Sum of squared distances (SSE)')
    plt.title('Elbow Method for Optimal K')
    plt.show()

    # 手动选择最佳的 K 值
    best_k = int(input("请输入最佳的 K 值: "))
    return best_k

# 评估聚类结果
def evaluate_clustering(features, labels):
    if len(set(labels)) > 1:  # 至少有两个簇
        score = silhouette_score(features, labels)
        print(f"Silhouette Score: {score:.4f}")
        return score
    else:
        print("所有样本都在一个簇中，无法计算轮廓系数。")
        return None

# 保存聚类结果到 CSV 文件
def save_cluster_results_to_csv(image_paths, labels, output_csv):
    df = pd.DataFrame({
        'image_path': image_paths,
        'cluster': labels
    })
    df.to_csv(output_csv, index=False)
    print(f"聚类结果已保存到 {output_csv}")

# 可视化聚类结果
def visualize_clusters(image_paths, labels, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        label = labels[i]
        output_path = os.path.join(output_dir, f"cluster_{label}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cv2.imwrite(os.path.join(output_path, os.path.basename(img_path)), img)

# 主函数
def main():
    # 设置图片文件夹路径和输出路径
    image_dir = r'D:\lu\luone\juleijiaohdbscan\haopicture'  # 替换为您的图片文件夹路径
    output_dir = r'D:\lu\luone\juleijiaohdbscan\TRAN'  # 替换为保存聚类结果的文件夹路径
    output_csv = r'D:\lu\luone\juleijiaohdbscan\result.csv'  # 确保是一个文件路径，而不是文件夹路径

    # 加载数据集
    dataset = ImageDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 初始化预训练 ResNet 特征提取模型
    model = PretrainedResNetExtractor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    features = []
    image_paths = []

    # 提取每张图片的特征
    with torch.no_grad():
        for images, img_names in dataloader:
            images = images.to(device)
            outputs = model(images)  # 提取特征
            features.append(outputs.cpu().numpy())
            image_paths.extend([os.path.join(image_dir, img_name) for img_name in img_names])

    features = np.vstack(features)  # 将所有特征合并成一个大的数组

    # 对特征进行标准化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 使用手肘法确定最佳的 K 值
    best_k = determine_best_k(features, max_k=10)

    # 使用最佳 K 值进行 KMeans 聚类
    kmeans = KMeans(n_clusters=best_k, random_state=0)
    labels = kmeans.fit_predict(features)

    # 评估聚类效果
    evaluate_clustering(features, labels)

    # 保存聚类结果到 CSV
    save_cluster_results_to_csv(image_paths, labels, output_csv)

    # 可视化聚类结果
    visualize_clusters(image_paths, labels, output_dir)

if __name__ == "__main__":
    main()
