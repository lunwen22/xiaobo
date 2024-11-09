import os
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取 ini 文件数据
def load_data_from_ini(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    data = np.array([float(line.strip()) for line in data if line.strip()])
    return data

# 绘制并保存图像
def plot_and_save(signal, file_name, output_folder):
    # 创建图像
    plt.figure(figsize=(10, 6))

    # 绘制原始信号
    plt.plot(signal)
    plt.title(f"自适应小波阈值去噪信号")

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 保存图像
    output_path = os.path.join(output_folder, f"{file_name}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"图像已保存到: {output_path}")

# 处理单个 ini 文件并生成图像
def process_ini_files(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".ini"):
                file_path = os.path.join(root, file_name)
                signal = load_data_from_ini(file_path)
                plot_and_save(signal, file_name.replace(".ini", ""), output_folder)

# 主程序入口
if __name__ == "__main__":
    input_folder = r"D:\shudeng\波形图\聚类优化\处理后的数据\分类结果\中深井"  # 输入文件夹路径
    output_folder = r"D:\shudeng\波形图\聚类优化\处理后的数据\分类结果1\中深井"  # 输出文件夹路径
    
    process_ini_files(input_folder, output_folder)
