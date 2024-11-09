import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 假设声波速度和采样频率
v_sound = 1500  # 声波在油井中的传播速度 (m/s)
fs = 1000       # 采样频率 (Hz)

# 读取ini格式文件数据
def load_data_from_ini(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    data = np.array([float(line.strip()) for line in data if line.strip()])
    return data

# 液面深度计算函数
def calculate_well_depth(first_valley, fs=1000, v_sound=1500):
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

# 自动创建不存在的文件夹
def create_directory_if_not_exists(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
    except OSError as error:
        print(f"创建文件夹 {folder_path} 失败，错误信息: {error}")

# 保存液面深度信息函数
def save_depth_info(well_type, file_name, depth, output_folder):
    # 根据井类型创建文件夹路径
    folder_path = os.path.join(output_folder, well_type)
    create_directory_if_not_exists(folder_path)

    # 构建输出文件路径
    output_file = os.path.join(folder_path, f"{file_name}_depth_info.txt")

    # 将井深度信息保存到文件
    with open(output_file, 'w') as f:
        f.write(f"液面深度: {depth:.2f} 米\n")
        f.write(f"井类型: {well_type}\n")
    
    print(f"文件 {output_file} 保存成功。")

# 处理所有 ini 文件并进行分类
def process_ini_files(ini_folder, output_folder):
    for root, _, files in os.walk(ini_folder):
        for file_name in files:
            if file_name.endswith('.ini'):
                file_path = os.path.join(root, file_name)
                signal = load_data_from_ini(file_path)

                # 找到液面最显著点（解包三个值）
                significant_point, _, _ = find_significant_point(signal)
                if significant_point is not None:
                    depth = calculate_well_depth(significant_point, fs, v_sound)
                    well_type = classify_well_depth(depth)

                    # 保存井深度信息
                    save_depth_info(well_type, file_name, depth, output_folder)
                else:
                    print(f"{file_name} 没有检测到显著的液面点")


# 主程序入口
if __name__ == "__main__":
    ini_folder = r"D:\shudeng\波形图\聚类优化\处理后的数据"  # 输入文件夹路径
    output_folder = r"D:\shudeng\波形图\聚类优化\处理后的数据\液面深度分类"  # 输出文件夹路径
    process_ini_files(ini_folder, output_folder)
