import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Bidirectional # type: ignore
from tensorflow.keras.layers import InputLayer # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # type: ignore
from scipy.signal import savgol_filter

# 读取数据文件
def read_data_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".ini"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as file:
                    signal = np.array([float(line.strip()) for line in file.readlines()])
                    data.append((filename, signal))
                    print(f"Successfully read {filename}, signal length: {len(signal)}")  # Debug output
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return data

# 信号预处理
def preprocess_signal(signal):
    signal = (signal - np.mean(signal)) / np.std(signal)
    trend = savgol_filter(signal, window_length=51, polyorder=2)
    detrended_signal = signal - trend
    smoothed_signal = savgol_filter(detrended_signal, window_length=11, polyorder=2)
    return smoothed_signal

# CNN去噪
def cnn_denoise(signal):
    model = Sequential([
        InputLayer(input_shape=(len(signal), 1, 1)),
        Conv2D(32, (5, 1), activation='relu', padding='same'),
        MaxPooling2D((2, 1)),
        Conv2D(64, (5, 1), activation='relu', padding='same'),
        Flatten(),
        Dense(len(signal))
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    signal = signal.reshape((1, len(signal), 1, 1))
    model.fit(signal, signal, epochs=200, batch_size=32, verbose=0)
    denoised_signal = model.predict(signal).flatten()
    return denoised_signal

# LSTM去噪
def lstm_predict(signal):
    num_features = 1
    signal = signal.reshape(-1, 1)
    generator = TimeseriesGenerator(signal, signal, length=1, batch_size=1)
    
    model = Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=(None, num_features)),
        Bidirectional(LSTM(50, activation='relu', return_sequences=True)),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(generator, epochs=200, verbose=0)
    denoised_signal = model.predict(signal).flatten()
    return denoised_signal

# 组合去噪方法
def combine_methods(signal1, signal2):
    combined_signal = (signal1 + signal2) / 2
    return combined_signal

# 主函数
def process_signals(directory):
    data_files = read_data_files(directory)
    
    if not data_files:
        print('No data files found.')
        return
    
    t = np.linspace(0, 10, len(data_files[0][1]))  # 假设时间序列为0到10

    for filename, signal in data_files:
        plt.figure(figsize=(12, 10))
        
        plt.subplot(5, 1, 1)
        plt.plot(t, signal)
        plt.title(f'Original Signal - {filename}')
        
        preprocessed_signal = preprocess_signal(signal)
        plt.subplot(5, 1, 2)
        plt.plot(t, preprocessed_signal)
        plt.title(f'Preprocessed Signal - {filename}')
        
        denoised_signal_cnn = cnn_denoise(preprocessed_signal)
        plt.subplot(5, 1, 3)
        plt.plot(t, denoised_signal_cnn)
        plt.title(f'Denoised Signal using CNN - {filename}')
        
        denoised_signal_lstm = lstm_predict(preprocessed_signal)
        plt.subplot(5, 1, 4)
        plt.plot(t, denoised_signal_lstm)
        plt.title(f'Denoised Signal using LSTM - {filename}')
        
        combined_denoised_signal = combine_methods(denoised_signal_cnn, denoised_signal_lstm)
        plt.subplot(5, 1, 5)
        plt.plot(t, combined_denoised_signal)
        plt.title(f'Combined Denoised Signal - {filename}')
        
        plt.tight_layout()
        plt.show()

# 运行主函数
if __name__ == "__main__":
    directory = r"D:\shudeng\ProofingTool\数据"  # 设置数据路径
    process_signals(directory)
