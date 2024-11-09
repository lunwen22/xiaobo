import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GRU, TimeDistributed, Dropout, LayerNormalization, Input, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

# 设置 GPU 内存增长
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 自定义层，用于包装 tf.expand_dims
class ExpandDimsLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=1)

# 数据加载与预处理（多线程）
def load_images_from_folder(folder, img_size):
    def process_image(filename):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                return np.expand_dims(img, axis=-1) / 255.0
        return None

    with ThreadPoolExecutor() as executor:
        images = list(executor.map(process_image, sorted(os.listdir(folder))))
        images = [img for img in images if img is not None]
    return np.array(images)

# 优化后的 CNN + Transformer 模型
def build_cnn_transformer(input_shape, num_heads=6, gru_units=256, learning_rate=1e-5, dropout_rate=0.3):
    inputs = Input(shape=input_shape)

    # CNN 层
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(inputs)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Flatten())(x)

    # GRU 层
    x = GRU(gru_units, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)

    # 多头自注意力层
    transformer_input = ExpandDimsLayer()(x)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=gru_units)(transformer_input, transformer_input)
    x = Dropout(dropout_rate)(attn_output)
    x = tf.keras.layers.Add()([x, transformer_input])
    x = LayerNormalization()(x)

    # FFN部分
    ffn_output = Dense(gru_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    x = tf.keras.layers.Add()([x, ffn_output])
    x = LayerNormalization()(x)

    # 输出层
    x = Flatten()(x)
    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# 文件夹路径和图像大小
data_folder = r"D:\lu\luone\juleijiaohdbscan\haochulihoushuju"
image_folder = r"D:\lu\luone\juleijiaohdbscan\haopicture"
output_folder = r"D:\lu\luone\CNN+Transformer3"
img_size = (64, 64)
sequence_length = 10

# 检查并创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 加载图像数据（多线程加速）
X = load_images_from_folder(image_folder, img_size)
X = X[:(X.shape[0] // sequence_length) * sequence_length]
X = X.reshape((-1, sequence_length, img_size[0], img_size[1], 1))

# 创建模拟标签并归一化
y = np.random.randint(500, 5500, size=(X.shape[0],))
y = (y - 500) / (5500 - 500)

# 数据拆分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 tf.data 数据集格式
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 构建模型
input_shape = (sequence_length, img_size[0], img_size[1], 1)
model = build_cnn_transformer(input_shape)

# 模型训练，调整学习率和早停条件
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)

history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[early_stopping, reduce_lr])

# 保存预测结果与分析
X_test = X.copy()
predicted_points = model.predict(X_test, batch_size=batch_size)
real_values = np.random.randint(500, 5500, size=(X_test.shape[0],))

results = []
for i in range(len(predicted_points)):
    predicted_value = int(predicted_points[i][0] * (5500 - 500) + 500)
    real_value = real_values[i]
    error = abs(predicted_value - real_value)
    results.append([f"image_{i}.png", real_value, predicted_value, error])

# 保存为 CSV 文件
results_df = pd.DataFrame(results, columns=['Filename', 'Real Value', 'Predicted Value', 'Error'])
csv_output_path = os.path.join(output_folder, 'prediction_results.csv')
results_df.to_csv(csv_output_path, index=False)

print(f"预测结果已保存到 {csv_output_path}")
