import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy.signal import welch, find_peaks, savgol_filter
from scipy.stats import kurtosis, skew

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 假设声波速度和采样频率
v_sound = 350  # 声波在油井中的传播速度 (m/s)
fs = 5120       # 采样频率 (Hz)

# 读取ini格式文件数据
def load_data_from_ini(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    data = np.array([float(line.strip()) for line in data if line.strip()])
    return data

# 液面深度计算函数
def calculate_well_depth(first_valley, fs=1000, v_sound=350):
    time_delay = first_valley / fs  # 计算时间延迟
    depth = (v_sound * time_delay) / 2  # 往返时间除以2得到深度
    return depth

# 寻找信号中的所有凹面点和凸面点，并找到最显著点
def find_significant_point(signal, min_index=1000):
    # 动态调整平滑窗口的长度，确保窗口不会超过信号的长度
    window_length = min(51, len(signal))  # 51 是之前设定的窗口长度，如果信号长度小于 51，使用信号长度
    if window_length % 2 == 0:
        window_length -= 1  # Savitzky-Golay 滤波器要求 window_length 是奇数

    # 对信号进行平滑处理，忽略较小的波动
    smoothed_signal = savgol_filter(signal, window_length, 3)

    # 反转信号以便检测凹面点
    inverted_signal = -smoothed_signal

    # 找到所有凹面点和凸面点，设定最小距离，忽略早期较小波动
    valleys, _ = find_peaks(inverted_signal[min_index:], distance=200)  # 找到凹面点
    peaks, _ = find_peaks(smoothed_signal[min_index:], distance=200)  # 找到凸面点

    # 将valleys和peaks重新映射回原始信号的索引
    valleys = valleys + min_index
    peaks = peaks + min_index

    # 选择最显著的凹面点和凸面点
    if len(valleys) > 0 and len(peaks) > 0:
        most_significant_valley = valleys[np.argmin(smoothed_signal[valleys])]  # 找到最显著的凹面点
        most_significant_peak = peaks[np.argmax(smoothed_signal[peaks])]  # 找到最显著的凸面点

        # 选择其中绝对值更大的点作为最显著点
        if abs(smoothed_signal[most_significant_valley]) > abs(smoothed_signal[most_significant_peak]):
            return most_significant_valley, smoothed_signal[most_significant_valley], 'valley'
        else:
            return most_significant_peak, smoothed_signal[most_significant_peak], 'peak'
    elif len(valleys) > 0:
        most_significant_valley = valleys[np.argmin(smoothed_signal[valleys])]
        return most_significant_valley, smoothed_signal[most_significant_valley], 'valley'
    elif len(peaks) > 0:
        most_significant_peak = peaks[np.argmax(smoothed_signal[peaks])]
        return most_significant_peak, smoothed_signal[most_significant_peak], 'peak'
    else:
        return None, None, 'unknown'


# 液面信号的转换（检测液面点）
def detect_well_depth_by_peaks(signal, fs=1000, v_sound=1500):
    # 反转信号以便检测凹面点
    inverted_signal = -signal
    # 找到凹面点
    valleys, _ = find_peaks(inverted_signal, distance=200)

    # 如果找到了凹面点，选择最显著的凹面点计算深度
    if len(valleys) > 0:
        most_significant_valley = valleys[np.argmin(signal[valleys])]  # 找到最显著的凹面点
        depth = calculate_well_depth(most_significant_valley, fs, v_sound)  # 计算深度
        
        # 确保 depth 是一个浮点数，而不是序列或其他类型
        if isinstance(depth, (list, np.ndarray)):
            depth = depth[0]  # 如果是序列，取第一个元素
        return float(depth), "井类型", most_significant_valley
    else:
        return None, "未知", None



# 井深度分类函数
def classify_well_depth(depth):
    if depth <= 2000:
        return "浅井"
    elif depth <= 4500:
        return "中深井"
    elif depth <= 6000:
        return "深井"
    else:
        return "超深井"

# 噪声类型分类，基于频率能量分布
def classify_noise_by_frequency(signal, fs=1000):
    freqs, psd = welch(signal, fs=fs, nperseg=min(1024, len(signal)))
    
    # 计算各个频段的能量
    low_freq_energy = np.mean(psd[(freqs >= 0) & (freqs <= 100)])  # 低频：0-100Hz
    mid_freq_energy = np.mean(psd[(freqs > 100) & (freqs <= 300)])  # 中频：100-300Hz
    high_freq_energy = np.mean(psd[freqs > 300])  # 高频：300Hz 以上
    
    # 噪声类型分类函数，根据低频、中频和高频能量进行分类
def classify_noise_type(low_freq_energy, mid_freq_energy, high_freq_energy, low_freq_threshold, mid_freq_threshold, high_freq_threshold):
    if high_freq_energy > high_freq_threshold:
        return "高频噪声"
    elif mid_freq_energy > mid_freq_threshold:
        return "中频噪声"
    else:
        return "低频噪声"

# 提取信号特征，加入ZCR和方差
def extract_signal_features(signal, fs=1000):
    amplitude = np.ptp(signal)  # 振幅
    std_dev = np.std(signal)    # 标准差
    energy = np.sum(signal**2)  # 能量
    peaks = len(find_peaks(signal)[0])  # 峰值计数

    skewness = skew(signal)     # 偏度
    kurt = kurtosis(signal)     # 峰度
    variance = np.var(signal)   # 方差

    freqs, psd = welch(signal, fs=fs, nperseg=min(1024, len(signal)))
    
    # 将 nperseg 动态调整为信号长度
    nperseg = min(1024, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    
    low_freq_energy = np.mean(psd[(freqs >= 0) & (freqs <= 100)])  # 低频：0-100Hz
    mid_freq_energy = np.mean(psd[(freqs > 100) & (freqs <= 300)])  # 中频：100-300Hz
    high_freq_energy = np.mean(psd[freqs > 300])  # 高频：300Hz 以上

    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))  # ZCR

    return [amplitude, std_dev, energy, peaks, low_freq_energy, mid_freq_energy, high_freq_energy, zcr, variance]

# 液面值转换函数
def liquid_change(liquid, sonic):
    # 确保 liquid 是单个浮点数而不是序列
    if isinstance(liquid, (list, np.ndarray)):
        liquid = liquid[0]  # 如果 liquid 是序列或数组，取第一个元素
    
    # 检查是否是数值类型
    if isinstance(liquid, (int, float)):
        real_liquid = liquid / 470 / 2 * sonic  # 根据 sonic 值转换液面深度
        return real_liquid
    else:
        raise TypeError(f"liquid 不是数值类型，而是 {type(liquid)}，值为: {liquid}")

# 自动创建不存在的文件夹
def create_directory_if_not_exists(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
    except OSError as error:
        print(f"创建文件夹 {folder_path} 失败，错误信息: {error}")
        
# 根据噪音类型创建文件夹并保存频谱图
def save_spectrum_image(noise_type, file_name, freqs, psd, output_folder):
    # 创建噪声类型文件夹路径
    folder_path = os.path.join(output_folder, noise_type)
    create_directory_if_not_exists(folder_path)

    # 生成频谱图像
    plt.figure()
    plt.plot(freqs, 10 * np.log10(psd))  # 使用dB显示频谱
    plt.title(f"{file_name} - {noise_type} 频谱")
    plt.xlabel("频率 (Hz)")
    plt.ylabel("功率谱密度 (dB/Hz)")

    # 保存图像
    output_image_path = os.path.join(folder_path, f"{file_name}_spectrum.png")
    plt.savefig(output_image_path)
    plt.close()
    
    print(f"频谱图 {output_image_path} 保存成功。")
    
# 自适应计算频率阈值，确保每个频段都有噪音分类，基于ZCR调整
def calculate_adaptive_frequency_thresholds(signal, fs=1000):
    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    
    # 计算ZCR并用于动态调整阈值
    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))

    # 动态调整的频率阈值，根据ZCR调节敏感度
    low_freq_threshold = np.mean(psd[(freqs >= 0) & (freqs <= 100)]) * (1 + zcr)
    mid_freq_threshold = np.mean(psd[(freqs > 100) & (freqs <= 300)]) * (1 + zcr)
    high_freq_threshold = np.mean(psd[freqs > 300]) * (1 + zcr)

    return low_freq_threshold, mid_freq_threshold, high_freq_threshold

# 根据噪音类型创建文件夹并保存
def save_to_noise_type_folder(noise_type, file_name, depth, significant_point, significant_value, point_type, output_folder):
    # 添加调试信息，便于检查保存路径及分类信息
    print(f"保存文件至: {noise_type} 文件夹, 深度: {depth}, 井类型: {classify_well_depth(depth)}")

    # 根据噪音类型创建文件夹路径
    folder_path = os.path.join(output_folder, noise_type)  # 使用正确的变量名 folder_path
    create_directory_if_not_exists(folder_path)  # 确保文件夹存在

    # 构建输出文件路径
    output_file = os.path.join(folder_path, f"{file_name}_info.txt")

    # 将结果保存到指定文件
    with open(output_file, 'w') as f:
        f.write(f"最显著点位置: {significant_point}\n")
        f.write(f"最显著点数据: {significant_value}\n")
        f.write(f"液面深度: {depth}\n")
        f.write(f"井类型: {classify_well_depth(depth)}\n")
        f.write(f"点类型: {point_type}\n")

    # 提示成功保存信息
    print(f"文件 {output_file} 保存成功。")

        
# 自适应计算频率阈值，确保每个频段都有噪音分类，基于ZCR调整
def calculate_adaptive_frequency_thresholds(signal, fs=1000):
    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    
    # 计算ZCR并用于动态调整阈值
    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))

    # 动态调整的频率阈值，根据ZCR调节敏感度
    low_freq_threshold = np.mean(psd[(freqs >= 0) & (freqs <= 100)]) * (1 + zcr)
    mid_freq_threshold = np.mean(psd[(freqs > 100) & (freqs <= 300)]) * (1 + zcr)
    high_freq_threshold = np.mean(psd[freqs > 300]) * (1 + zcr)

    return low_freq_threshold, mid_freq_threshold, high_freq_threshold

# BIC 自动确定最佳 K 值进行聚类，增加 reg_covar 参数
def perform_clustering_with_bic(features, max_clusters=10, reg_covar=1e-6):
    bic_scores = []
    gmm_models = []

    for k in range(1, max_clusters + 1):
        try:
            # 增加 reg_covar 参数来正则化协方差矩阵
            gmm = GaussianMixture(n_components=k, random_state=42, reg_covar=reg_covar)
            gmm.fit(features)
            bic_scores.append(gmm.bic(features))
            gmm_models.append(gmm)
        except ValueError:
            print(f"聚类数 {k} 时 GMM 拟合失败，跳过。")
            continue

    # 确保至少有一个模型成功拟合
    if len(bic_scores) == 0:
        raise ValueError("没有成功拟合的 GMM 模型，请调整 max_clusters 或 reg_covar 参数。")

    best_k = np.argmin(bic_scores) + 1
    best_gmm = gmm_models[np.argmin(bic_scores)]
    print(f"根据 BIC，最佳聚类数 (K) 为: {best_k}")

    # 使用最佳 GMM 进行聚类
    labels = best_gmm.fit_predict(features)
    return labels, best_gmm

# 给每个类别打标签
def label_clusters(features, labels, depth_list, low_freq_threshold, mid_freq_threshold, high_freq_threshold):
    labeled_clusters = {}
    
    for label in np.unique(labels):
        cluster_data = features[labels == label]
        cluster_depth = depth_list[labels == label]
        
        avg_low_freq_energy = np.mean([data[4] for data in cluster_data])
        avg_mid_freq_energy = np.mean([data[5] for data in cluster_data])
        avg_high_freq_energy = np.mean([data[6] for data in cluster_data])
        variance = np.mean([data[8] for data in cluster_data])  # 方差
        
        # 根据噪声频率类型和方差
        noise_type = classify_noise_type(avg_low_freq_energy, avg_mid_freq_energy, avg_high_freq_energy, low_freq_threshold, mid_freq_threshold, high_freq_threshold)

        # 根据井深分类
        avg_depth = np.mean(cluster_depth)
        well_type = classify_well_depth(avg_depth)
        
        labeled_clusters[label] = f"{noise_type}{well_type}"
    
    return labeled_clusters

# 使用 PCA 技术进行降维可视化，并标注噪声和井深分类
def visualize_with_pca_and_labels(features, labels, labeled_clusters, output_folder):
    pca = PCA(n_components=2)  # 将特征降到2维
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_points = reduced_features[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"{labeled_clusters[label]} (Cluster {label})")
    
    plt.title("PCA 降维后的聚类结果（带类型标注）")
    plt.xlabel("主成分 1")
    plt.ylabel("主成分 2")
    plt.legend()

    create_directory_if_not_exists(output_folder)
    plt.savefig(os.path.join(output_folder, "PCA_聚类可视化_带标签.png"))
    plt.close()

    print("PCA 聚类可视化图片保存成功，带有类型标注。")

# 主程序入口
def process_ini_files_with_combined_method(ini_folder, output_folder, sonic=1500):
    file_names = []
    signals = []
    error_files = []  # 用于记录出错的文件
    
    # 遍历文件夹中的所有文件，读取数据
    for root, _, files in os.walk(ini_folder):
        for file_name in files:
            if file_name.endswith('.ini'):
                file_path = os.path.join(root, file_name)
                try:
                    signal = load_data_from_ini(file_path)
                    signals.append(signal)  # 将信号保存起来
                    file_names.append(file_name)
                    
                    # 基于频谱分析噪音类型
                    noise_type, freqs, psd = classify_noise_by_frequency(signal, fs)
                    
                    # 保存频谱图像
                    save_spectrum_image(noise_type, file_name, freqs, psd, output_folder)
                    
                except Exception as e:
                    print(f"处理文件 {file_name} 时发生错误: {e}")
                    error_files.append(file_name)
                    
      
                    # 对每个信号找到最显著点位置
                    significant_point, significant_value, point_type = find_significant_point(signal)

                    if significant_point is not None:
                        depth = calculate_well_depth(significant_point, fs, v_sound)
                        well_type = classify_well_depth(depth)
                        print(f"文件: {file_name} | 最显著点位置: {significant_point} | 最显著点数据: {significant_value} | 液面深度: {depth:.2f} 米 | 井类型: {well_type} | 点类型: {point_type}")

                        # 转换液面深度
                        try:
                            depth = liquid_change(depth, sonic)
                        except TypeError as e:
                            print(f"处理文件 {file_name} 时出现错误: {e}")
                            error_files.append(file_name)
                            continue  # 跳过有错误的文件

                        # 根据频谱能量进行噪音分类
                        freqs, psd = welch(signal, fs=fs, nperseg=1024)
                        
                        # 计算不同频率区间的能量
                        low_freq_energy = np.mean(psd[(freqs >= 0) & (freqs <= 100)])
                        mid_freq_energy = np.mean(psd[(freqs > 100) & (freqs <= 300)])
                        high_freq_energy = np.mean(psd[freqs > 300])
                        
                        # 自适应阈值可以根据信号的 ZCR 调整
                        low_freq_threshold, mid_freq_threshold, high_freq_threshold = calculate_adaptive_frequency_thresholds(signal)
                        
                        # 添加噪音分类逻辑
                        if high_freq_energy > high_freq_threshold:
                            noise_type = "高频噪声"
                        elif mid_freq_energy > mid_freq_threshold:
                            noise_type = "中频噪声"
                        else:
                            noise_type = "低频噪声"

                        # 保存到相应的文件夹
                        save_to_noise_type_folder(noise_type, file_name, depth, significant_point, significant_value, point_type, output_folder)
                    else:
                        print(f"文件: {file_name} | 未检测到显著的点")
                except Exception as e:
                    print(f"处理文件 {file_name} 时发生错误: {e}")
                    error_files.append(file_name)

    # 打印错误文件列表
    if error_files:
        print(f"以下文件在处理过程中出现错误: {error_files}")
    else:
        print("所有文件均成功处理。")

    # 提取特征
    features = [extract_signal_features(signal) for signal in signals]
    features = np.array(features)
    if features.ndim == 1:
        features = features.reshape(-1, 1)  # 如果是1D数组，转换为2D数组

    # 处理缺失值
    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)

    # 检测井深度并计算
    depth_list = []
    for signal, file_name in zip(signals, file_names):
        # 这里也解包三个值
        depth, well_type, valley_point = detect_well_depth_by_peaks(signal)

        # 输出调试信息
        print(f"调试信息: signal = {signal[:10]}, depth = {depth}, 类型 = {type(depth)}")

        if depth is not None:
            # 确保 depth 是一个数字，而不是序列
            if isinstance(depth, (list, np.ndarray)):
                depth = depth[0]  # 如果是序列，取第一个元素
                print(f"depth 是一个序列，取第一个元素: {depth}")
            
            try:
                depth = liquid_change(depth, sonic)
                print(f"液面深度转换成功: {depth}")
                depth_list.append(depth)
            except TypeError as e:
                print(f"处理文件 {file_name} 时，转换液面深度出现错误: {e}, depth 值: {depth}, 类型: {type(depth)}")
                depth_list.append(0)
        else:
            depth_list.append(0)

    # 执行聚类
    labels, _ = perform_clustering_with_bic(features)

    # 自适应频率阈值计算
    low_freq_threshold, mid_freq_threshold, high_freq_threshold = calculate_adaptive_frequency_thresholds(signals[0])

    # 给每个聚类打标签
    labeled_clusters = label_clusters(features, labels, np.array(depth_list), low_freq_threshold, mid_freq_threshold, high_freq_threshold)

    # 使用PCA技术进行可视化，带有类型标注
    visualize_with_pca_and_labels(features, labels, labeled_clusters, output_folder)


# 运行程序
if __name__ == "__main__":
    ini_folder = r"D:\shudeng\波形图\聚类优化\处理后的数据"  # 修改为实际的文件夹路径
    output_folder = r"D:\shudeng\波形图\聚类优化\处理后的数据\分类结果"  # 修改为实际的输出文件夹路径
    process_ini_files_with_combined_method(ini_folder, output_folder)

