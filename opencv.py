import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from ultralytics import YOLO
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import concurrent.futures
import sys
print(sys.executable)

# YOLO 模型加载
yolo_model = YOLO('yolov5s.pt')

# 液面高度转换公式
def liquid_change(liquid_pixel, image_height, max_depth):
    return (liquid_pixel / image_height) * max_depth

# 自动识别液面波动点的函数，动态调整阈值
def detect_fluctuation_point(signal, log_file, filename, min_pixel=100):
    try:
        # 记录并显示截取的信号
        log_file.write(f'{filename} - 截取后的信号（从{min_pixel}像素开始）: {signal[min_pixel:]}\n')
        plt.plot(signal)
        plt.title(f'Signal for {filename}')
        plt.show()

        smoothed_signal = signal
        diff_signal = np.diff(smoothed_signal)
        
        # 根据信号动态调整阈值
        threshold = np.mean(np.abs(diff_signal)) + np.std(diff_signal) * 0.5
        log_file.write(f'{filename} - 动态调整的阈值: {threshold}\n')

        # 检测波峰，调整波峰距离参数
        peaks, _ = find_peaks(smoothed_signal, height=threshold, distance=30)
        log_file.write(f'{filename} - 检测到的波峰位置: {peaks}\n')

        valid_peaks = peaks[peaks >= min_pixel]
        log_file.write(f'{filename} - 筛选后的波峰位置（像素大于{min_pixel}）：{valid_peaks}\n')

        if len(valid_peaks) > 0:
            highest_peak = valid_peaks[np.argmax(smoothed_signal[valid_peaks])]
            return highest_peak
        else:
            return None
    except Exception as e:
        log_file.write(f'{filename} - 检测波动点时出错: {str(e)}\n')
        return None

# LSTM 数据准备
def prepare_lstm_data(signal, n_steps):
    X, y = [], []
    for i in range(len(signal)):
        end_ix = i + n_steps
        if end_ix > len(signal) - 1:
            break
        seq_x, seq_y = signal[i:end_ix], signal[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# LSTM 模型构建与训练
def train_lstm_model(X_train, y_train, n_steps):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=200, verbose=0)
    return model

# 处理图像并应用 YOLO 检测与 LSTM 预测，保存处理后的图像
def process_image_with_yolo(image_path, output_folder):
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    if image.shape[-1] == 4:  # 检查是否为4通道图像
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # YOLO 模型进行检测
    results = yolo_model(image)
    results[0].plot()  # 可视化检测结果

    # 提取液面信号
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    signal = np.mean(gray, axis=1)

    # 使用 LSTM 预测
    n_steps = 50
    X_train, y_train = prepare_lstm_data(signal, n_steps)
    lstm_model = train_lstm_model(X_train, y_train, n_steps)
    yhat = lstm_model.predict(X_train)

    # 检测波动点并计算液面高度
    fluctuation_index = detect_fluctuation_point(signal, open('log.txt', 'a'), image_path)
    if fluctuation_index is not None:
        liquid_height_pixel = fluctuation_index
        image_height = gray.shape[0]
        real_liquid_height = liquid_change(liquid_height_pixel, image_height, 100)
        print(f'实际液面高度：{real_liquid_height:.2f} 米')

    # 保存处理后的图像
    processed_image_path = os.path.join(output_folder, f"processed_{os.path.basename(image_path)}")
    cv2.imwrite(processed_image_path, image)
    print(f"处理后的图像已保存至: {processed_image_path}")

    cv2.imshow("YOLOv5 Detection", results[0].img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 多线程处理图像文件
def process_images_in_parallel(image_folder, output_folder):
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda path: process_image_with_yolo(path, output_folder), image_paths)

# 主程序：批量处理图像并结合 YOLO 和 LSTM 进行液面深度预测，并保存处理后的图像
image_folder = 'D:/shudeng/julei/dizhendang'
output_folder = 'D:/shudeng/julei/dizhendang/processed'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

process_images_in_parallel(image_folder, output_folder)
