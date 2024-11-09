import pywt
import numpy as np

import numpy as np
def denoise_evaluate(signal_pure, noise, signal_denoised):
    value = {}    # SNR计算
    p_signal = np.sum(signal_pure**2) / len(signal_pure)
    p_noise = np.sum(noise**2) / len(noise)
    SNR = 10 * np.log10(p_signal / p_noise)
    value['SNR'] = SNR
    MSE = np.mean((signal_pure - signal_denoised) ** 2)
    value['MSE'] = MSE
    # RMSE计算
    RMSE = np.sqrt(MSE)
    value['RMSE'] = RMSE
    return value
def denoised_with_pywt_universal(data, wavelet='db1', mode='soft', level=1):
    # 计算多级小波变换的系数
    coeffs = pywt.wavedec(data, wavelet, mode='symmetric', level=level)
    # 计算阈值
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    # 固定阈值去噪
    coeffs[1:] = (pywt.threshold(i, value=threshold, mode=mode) for i in coeffs[1:])
    # 重构信号
    return pywt.waverec(coeffs, wavelet, mode='symmetric')

import numpy as np
import pywt

def sure_threshold(coeffs):
    n = len(coeffs)
    risks = np.zeros(n)
    for i in range(n):
        threshold = coeffs[i]
        # 计算硬阈值处理后的风险
        risks[i] = n - 2 * np.sum(coeffs > threshold) + np.sum((coeffs - threshold)**2)
    # 选择最小风险对应的阈值
    min_risk_index = np.argmin(risks)
    return coeffs[min_risk_index]

def denoised_with_pywt_sure(data, wavelet='db1', mode='soft', level=1):
    # 计算多级小波变换的系数
    coeffs = pywt.wavedec(data, wavelet, mode='symmetric', level=level)
    # 对每一层系数应用SURE方法计算阈值并去噪
    for i in range(1, len(coeffs)):
        threshold = sure_threshold(np.abs(coeffs[i]))
        coeffs[i] = pywt.threshold(coeffs[i], value=threshold, mode=mode)
    # 重构信号
    return pywt.waverec(coeffs, wavelet, mode='symmetric')

import pywt
import numpy as np

def denoised_with_pywt_minimax(data, wavelet='db1', mode='soft', level=1):
    # 计算多级小波变换的系数
    coeffs = pywt.wavedec(data, wavelet, mode='symmetric', level=level)
    # 计算阈值
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    if len(data) >= 32:
        threshold = sigma * (0.3936+(0.1829*np.log(len(data))/np.log(2)))
    else:
        threshold = 0
    # 固定阈值去噪
    coeffs[1:] = (pywt.threshold(i, value=threshold, mode=mode) for i in coeffs[1:])
    # 重构信号
    return pywt.waverec(coeffs, wavelet, mode='symmetric')


import matplotlib.pyplot as plt
import matplotlib
# 创建一个信号

data = np.loadtxt("C:\\Users\\15849\\Desktop\\总\\126331124553553_OriData_20230720210920.ini")
t = np.linspace(0, len(data)-1, len(data), endpoint=False)
y = data

denoised_data_universal = denoised_with_pywt_universal(y, wavelet='db4', mode='soft', level=8)
denoised_data_sure = denoised_with_pywt_sure(y, wavelet='db4', mode='soft', level=8)
denoised_data_minimax = denoised_with_pywt_minimax(y, wavelet='db4', mode='soft', level=8)



plt.figure(1)
plt.plot(t, data, linewidth=1.5, color='b')
plt.title('pure signal', fontname='Times New Roman', fontsize=16)
plt.xlabel('Time', fontname='Times New Roman', fontsize=14)
plt.ylabel('Amplitude', fontname='Times New Roman', fontsize=14)

plt.figure(2)
plt.plot(t, denoised_data_universal, linewidth=1.5, color='r')
plt.title('fixed-threshold method', fontname='Simsun', fontsize=16)
plt.xlabel('Time', fontname='Times New Roman', fontsize=14)
plt.ylabel('Amplitude', fontname='Times New Roman', fontsize=14)
plt.figure(3)
plt.plot(t, denoised_data_sure, linewidth=1.5, color='g')
plt.title('Unbiased Likelihood Estimation Threshold', fontname='Simsun', fontsize=16)
plt.xlabel('Time', fontname='Times New Roman', fontsize=14)
plt.ylabel('Amplitude', fontname='Times New Roman', fontsize=14)
plt.figure(4)
plt.plot(denoised_data_minimax, linewidth=1.5, color='purple')
plt.title('extremum threshold', fontname='Simsun', fontsize=16)
plt.xlabel('Time', fontname='Times New Roman', fontsize=14)
plt.ylabel('Amplitude', fontname='Times New Roman', fontsize=14)
plt.show()
import os

# 导出降噪后的数据到CSV文件
output_folder = "C:\\Users\\86136\\Desktop\\denoised_data"
os.makedirs(output_folder, exist_ok=True)

np.savetxt(os.path.join(output_folder, 'denoised_data_universal.csv'), denoised_data_universal, delimiter=',')
np.savetxt(os.path.join(output_folder, 'denoised_data_sure.csv'), denoised_data_sure, delimiter=',')
np.savetxt(os.path.join(output_folder, 'denoised_data_minimax.csv'), denoised_data_minimax, delimiter=',')

# 计算并打印原始数据和处理后数据的统计信息
stats = {}

# 原始数据
stats['original'] = {
    'mean': np.mean(y),
    'variance': np.var(y),
    'std_dev': np.std(y)
}

# 固定阈值去噪数据
stats['universal'] = {
    'mean': np.mean(denoised_data_universal),
    'variance': np.var(denoised_data_universal),
    'std_dev': np.std(denoised_data_universal)
}

# 无偏去噪数据
stats['sure'] = {
    'mean': np.mean(denoised_data_sure),
    'variance': np.var(denoised_data_sure),
    'std_dev': np.std(denoised_data_sure)
}

# Minimax 阈值去噪数据
stats['minimax'] = {
    'mean': np.mean(denoised_data_minimax),
    'variance': np.var(denoised_data_minimax),
    'std_dev': np.std(denoised_data_minimax)
}

# 打印统计信息
for key, value in stats.items():
    print(f"{key} data:")
    print(f"Mean: {value['mean']:.4f}")
    print(f"Variance: {value['variance']:.4f}")
    print(f"Standard Deviation: {value['std_dev']:.4f}\n")