import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 读取图像
image_path = 'C:\\Users\\86136\\Desktop\\01b0bad3cc60c17c862c57db736edff.png'
image = Image.open(image_path)

# 创建图像绘制对象
fig, ax = plt.subplots(figsize=(12, 6))

# 显示图像
ax.imshow(image)
ax.set_title('Sound Wave Amplitude over Time')
ax.axis('off')

# 添加红色矩形框和注释
def add_annotation(ax, x_start, y_start, width, height, text, text_x, text_y):
    rect = patches.Rectangle((x_start, y_start), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(text_x, text_y, text, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# 根据图像比例添加红色矩形框和注释
add_annotation(ax, 50, 50, 100, 400, '起爆波', 160, 80)
add_annotation(ax, 700, 200, 200, 300, '接箍回波', 920, 230)
add_annotation(ax, 3000, 150, 200, 400, '液面回波', 3220, 180)

# 显示带标记的图像
plt.show()
