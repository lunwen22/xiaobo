import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
def set_chinese_font():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 预处理信号
def preprocess_signal(signal):
    # 可以加入滤波代码，这里简化处理
    return signal

# 分解信号成分（简化为示例）
def decompose_signal(signal):
    length = len(signal)
    component_1 = signal[:length//3]  # 起爆波
    component_2 = signal[length//3:2*length//3]  # 接箍回波
    component_3 = signal[2*length//3:]  # 液面回波
    return component_1, component_2, component_3

# 提取峰值作为信号特征
def extract_features(component):
    peaks, _ = find_peaks(component, height=0)
    return component[peaks]

# 对齐数据长度
def align_lengths(*arrays):
    max_length = max(len(arr) for arr in arrays)
    aligned_arrays = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=np.nan) for arr in arrays]
    return aligned_arrays

# 处理单个文件
def process_file(file_path, output_dir):
    data = np.loadtxt(file_path)
    processed_signal = preprocess_signal(data)
    
    # 分解信号
    comp1, comp2, comp3 = decompose_signal(processed_signal)
    
    # 提取特征
    features1 = extract_features(comp1)
    features2 = extract_features(comp2)
    features3 = extract_features(comp3)
    
    # 对齐数据长度
    features1, features2, features3 = align_lengths(features1, features2, features3)
    
    # 创建表格数据并保存
    table_data = pd.DataFrame({
        '起爆波': features1,
        '接箍回波': features2,
        '液面回波': features3
    })
    
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    table_data.to_csv(os.path.join(output_dir, f'{file_name}_extracted_features.csv'), index=False)
    
    # 绘制并保存分解后的信号
    set_chinese_font()
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(comp1)
    plt.title('起爆波')
    
    plt.subplot(3, 1, 2)
    plt.plot(comp2)
    plt.title('接箍回波')
    
    plt.subplot(3, 1, 3)
    plt.plot(comp3)
    plt.title('液面回波')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{file_name}_decomposed_signals.png'))
    plt.close()

# 主函数
def main(data_dir):
    output_dir = os.path.join(data_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path) and file_path.endswith('.ini'):
            process_file(file_path, output_dir)

if __name__ == "__main__":
    data_dir = 'D:\\shudeng\\ProofingTool\\数据'
    main(data_dir)
