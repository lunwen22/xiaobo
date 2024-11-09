import os
import shutil

# 定义路径
source_txt_folder = r"D:\shudeng\波形图\聚类优化\处理后的数据\液面深度分类"  # 含四个子文件夹的主目录
ini_folder = r"D:\shudeng\波形图\聚类优化\处理后的数据"  # 原始ini文件所在的目录
output_folder = r"D:\shudeng\波形图\聚类优化\处理后的数据\分类结果"  # 新的主文件夹路径

# 创建输出的主文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历source_txt_folder下的四个子文件夹
for category in os.listdir(source_txt_folder):
    category_path = os.path.join(source_txt_folder, category)
    
    if os.path.isdir(category_path):  # 确保是文件夹
        # 创建目标文件夹 (大文件夹下的四个子文件夹)
        output_category_folder = os.path.join(output_folder, category)
        os.makedirs(output_category_folder, exist_ok=True)
        
        # 遍历子文件夹中的所有.txt文件
        for txt_file in os.listdir(category_path):
            if txt_file.endswith('.txt'):
                # 获取.txt文件名，去掉扩展名
                ini_file_name = os.path.splitext(txt_file)[0]

                # 去掉文件名中的 "_depth_info" 部分
                ini_file_name = ini_file_name.replace("_depth_info", "")

                # 查找对应的 ini 文件
                ini_file_path = os.path.join(ini_folder, ini_file_name)
                
                # 检查是否对应的ini文件存在
                if os.path.exists(ini_file_path):
                    # 复制ini文件到新创建的子文件夹
                    shutil.copy(ini_file_path, output_category_folder)
                    print(f"已复制: {ini_file_name} 到 {output_category_folder}")
                else:
                    print(f"未找到对应的ini文件: {ini_file_name}")

print("所有文件复制完成。")
