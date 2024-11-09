import os
import numpy as np
import pywt
import pandas as pd

# 从ini文件中读取数据，假设每行都是一个数据点
def load_data_from_ini(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    data = np.array([float(line.strip()) for line in data if line.strip()])
    return data

# 定义SNR计算函数
def calculate_snr(original_signal, denoised_signal):
    noise = original_signal - denoised_signal
    snr = 10 * np.log10(np.sum(original_signal**2) / np.sum(noise**2))
    return snr

# 定义自适应小波阈值函数
def adaptive_wavelet_threshold(wavelet_coeffs, threshold, alpha):
    result_coeffs = np.sign(wavelet_coeffs) * np.maximum((np.abs(wavelet_coeffs) - threshold), 0) ** alpha
    return result_coeffs

# 自适应小波去噪函数
def adaptive_wavelet_denoising(signal, wavelet_name='coif5', alpha=1.0, level=5):
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    thresholded_coeffs = []
    
    for i in range(len(coeffs)):
        threshold = np.median(np.abs(coeffs[i])) / 0.6745
        thresholded_coeffs.append(adaptive_wavelet_threshold(coeffs[i], threshold, alpha))
    
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet_name)
    
    # 修正长度以匹配原始信号
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    else:
        denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), mode='constant')
    
    return denoised_signal

# 液面值转换（基于液面高度的计算公式）
def liquid_change(liquid, sonic):
    real_liquid = liquid / 470 / 2 * sonic  # 根据你之前的液面计算公式
    return real_liquid

# 使用SNR来选择最优alpha和最优层数的网格搜索
def grid_search_alpha_and_level(signal, original_signal, alpha_range=(0.5, 1.5), level_range=(2, 8), step=0.1):
    best_alpha = alpha_range[0]
    best_level = level_range[0]
    best_snr = -np.inf  # 初始化为负无穷大
    for level in range(level_range[0], level_range[1] + 1):
        for alpha in np.arange(alpha_range[0], alpha_range[1] + step, step):
            denoised_signal = adaptive_wavelet_denoising(signal, alpha=alpha, level=level)
            snr = calculate_snr(original_signal, denoised_signal)
            if snr > best_snr:
                best_snr = snr
                best_alpha = alpha
                best_level = level
    return best_alpha, best_level

# 处理文件夹中所有文件，并将数据保存到一个CSV中
def process_folder(input_folder, output_csv_file):
    all_data = []  # 用于存储所有文件的处理数据
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".ini"):
            file_path = os.path.join(input_folder, file_name)
            signal = load_data_from_ini(file_path)
            if len(signal) == 0:
                print(f"文件 {file_name} 为空，跳过处理。")
                continue
            
            # 将原始信号作为基准（没有去噪之前的信号）
            original_signal = signal.copy()
            
            # 寻找最佳alpha和层数
            best_alpha, best_level = grid_search_alpha_and_level(signal, original_signal)
            print(f"文件 {file_name} 的最佳 alpha 为: {best_alpha}, 最佳层数为: {best_level}")
            
            # 使用最优alpha和层数进行去噪
            denoised_signal = adaptive_wavelet_denoising(signal, alpha=best_alpha, level=best_level)

            # 计算液面高度
            for i, value in enumerate(denoised_signal):
                liquid_height = liquid_change(value, 343)  # 343 为空气中声速，按需修改
                
                all_data.append({
                    'id': i + 1,  # 假设用索引作为ID
                    '时间': '',  # 可根据需要填充时间
                    '设备号': file_name,  # 使用文件名作为设备号
                    'wb2nb1': '',  # 示例占位符，可以按需填写
                    '声速': value,  # 去噪后的信号值
                    '液面': liquid_height  # 液面高度
                })

    # 将数据写入CSV文件
    df = pd.DataFrame(all_data, columns=['id', '时间', '设备号', 'wb2nb1', '声速', '液面'])
    df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
    print(f"所有处理后的数据已保存至: {output_csv_file}")

# 主程序入口
if __name__ == "__main__":
    input_folder = r"D:\shudeng\ProofingTool\数据"
    output_csv_file = r"D:\shudeng\ProofingTool\数据\处理后的信号数据.csv"
    process_folder(input_folder, output_csv_file)
