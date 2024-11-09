import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GRU, TimeDistributed, Dropout, LayerNormalization, Input, MultiHeadAttention
from tensorflow.keras.models import Model

# 设置GPU内存增长
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} Physical GPUs detected")
    except RuntimeError as e:
        print(e)

# 自定义 ExpandDimsLayer，用于包装 tf.expand_dims
class ExpandDimsLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=1)

# 数据加载和预处理
def load_images_from_folder(folder, img_size):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load image: {filename}")
                continue
            img = cv2.resize(img, img_size)
            img = np.expand_dims(img, axis=-1) / 255.0
            images.append(img)
    return np.array(images)

# 构建 CNN + Transformer 模型
def build_cnn_transformer(input_shape):
    inputs = Input(shape=input_shape)

    # CNN 部分
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(inputs)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Flatten())(x)

    # GRU 部分
    x = GRU(128, return_sequences=True)(x)

    # Transformer 部分
    transformer_input = ExpandDimsLayer()(x)
    attn_output = MultiHeadAttention(num_heads=4, key_dim=128)(transformer_input, transformer_input)
    x = Dropout(0.1)(attn_output)
    x = tf.keras.layers.Add()([x, transformer_input])
    x = LayerNormalization()(x)

    # FFN部分 - 调整 FFN 输出的维度与前面的输入保持一致
    ffn_output = Dense(128, activation='relu')(x)
    ffn_output = Dropout(0.1)(ffn_output)
    x = tf.keras.layers.Add()([x, ffn_output])
    x = LayerNormalization()(x)

    # 输出层
    x = Flatten()(x)
    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    return model

# 设置文件夹路径和图像尺寸
data_folder = r"D:\lu\luone\juleijiaohdbscan\haochulihoushuju"
image_folder = r"D:\lu\luone\juleijiaohdbscan\haopicture"
output_folder = r"D:\lu\luone\CNN+Transformer2"
img_size = (64, 64)
sequence_length = 10

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 加载图像数据
X_train = load_images_from_folder(image_folder, img_size)

# 打印图片数量以检查是否合适
print("Loaded images shape:", X_train.shape)

# 修改图片数量，确保可以整除
num_images = X_train.shape[0]
X_train = X_train[:(num_images // sequence_length) * sequence_length]  # 丢弃不完整的部分

X_train = X_train.reshape((-1, sequence_length, img_size[0], img_size[1], 1))

# 生成模拟标签，并进行归一化处理
y_train = np.random.randint(500, 5500, size=(X_train.shape[0],))
y_train = (y_train - 500) / (5500 - 500)  # 归一化到 0-1 之间

# 构建模型
input_shape = (sequence_length, img_size[0], img_size[1], 1)
model = build_cnn_transformer(input_shape)
model.summary()

# 模型训练
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# 保存预测结果
X_test = load_images_from_folder(image_folder, img_size)
X_test = X_test[:(X_test.shape[0] // sequence_length) * sequence_length]  # 丢弃不完整的部分
X_test = X_test.reshape((-1, sequence_length, img_size[0], img_size[1], 1))
predicted_points = model.predict(X_test)
real_values = np.random.randint(500, 5500, size=(X_test.shape[0],))  # 模拟真实值
real_values_normalized = (real_values - 500) / (5500 - 500)  # 归一化
results = []

for i, filename in enumerate(sorted(os.listdir(image_folder))[:len(predicted_points)]):
    if filename.endswith(".png"):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)

        # 获取预测的深度点位置并反归一化
        predicted_value = int(predicted_points[i][0] * (5500 - 500) + 500)
        real_value = real_values[i]

        # 标记并保存图片
        cv2.putText(img, f"Predicted: {predicted_value}, Real: {real_value}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)

        # 保存预测值与真实值
        results.append([filename, real_value, predicted_value])

# 保存 CSV 文件
results_df = pd.DataFrame(results, columns=['Filename', 'Real Value', 'Predicted Value'])
csv_output_path = os.path.join(output_folder, 'prediction_results.csv')
results_df.to_csv(csv_output_path, index=False)

print(f"结果已保存到 {csv_output_path}")
