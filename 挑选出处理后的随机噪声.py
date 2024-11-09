import os
import shutil

# 定义文件夹路径
suiji_folder = r"D:\shudeng\波形图\聚类优化\suiji"  # 包含图片的文件夹
ini_data_folder = r"D:\shudeng\波形图\聚类优化\处理后的数据"  # 包含ini文件的文件夹
output_folder = r"D:\shudeng\波形图\聚类优化\suijizaoshengshuju"  # 自动创建的新文件夹

# 创建输出文件夹，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历 suiji 文件夹中的所有图片文件
for img_file in os.listdir(suiji_folder):
    # 检查文件扩展名是否为图片类型
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        # 提取图片文件名（去掉扩展名）
        img_name_without_ext = os.path.splitext(img_file)[0]
        
        # 在 ini_data_folder 中查找对应的 ini 文件
        corresponding_ini_file = img_name_without_ext + '.ini'
        ini_file_path = os.path.join(ini_data_folder, corresponding_ini_file)
        
        # 检查 ini 文件是否存在
        if os.path.exists(ini_file_path):
            # 将 ini 文件剪切到 output_folder
            shutil.move(ini_file_path, os.path.join(output_folder, corresponding_ini_file))
            print(f"文件 {corresponding_ini_file} 已剪切到 {output_folder}")
        else:
            print(f"未找到 {corresponding_ini_file} 对应的 ini 文件")

print("处理完成！")
