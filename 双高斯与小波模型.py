import os
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import pywt
import matplotlib.pyplot as plt
import warnings

# 定义双高斯函数
def double_gaussian(x, a1, b1, c1, a2, b2, c2):
    return a1 * np.exp(-((x - b1) ** 2) / (2 * c1 ** 2)) + a2 * np.exp(-((x - b2) ** 2) / (2 * c2 ** 2))

# 自适应小波去噪函数
def adaptive_thresholding(coeffs, method='sureshrink'):
    """应用自适应阈值法去噪"""
    thresholded_coeffs = []
    for coeff in coeffs:
        sigma = np.median(np.abs(coeff)) / 0.6745  # 估计噪声标准差
        if method == 'sureshrink':
            threshold = sigma * np.sqrt(2 * np.log(len(coeff)))
        else:
            raise ValueError(f"Unknown method: {method}")
        thresholded_coeffs.append(pywt.threshold(coeff, threshold, mode='soft'))
    return thresholded_coeffs

# 定义文件夹路径
input_folder = r'D:\shudeng\ProofingTool\数据'
output_folder = os.path.join(input_folder, '双高斯与小波模型结果')

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 忽略拟合中的警告
warnings.simplefilter('ignore', OptimizeWarning)

# 遍历所有 .ini 文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.ini'):
        # 完整文件路径
        file_path = os.path.join(input_folder, file_name)
        
        try:
            # 读取数据
            data = np.loadtxt(file_path)
            y_data = data  # 只有一列数据为测量值
            
            # 使用索引作为伪时间轴
            x_data = np.arange(len(y_data))

            # 双高斯拟合
            initial_guess = [1, len(x_data)/3, 1, 1, 2*len(x_data)/3, 1]  # 初始猜测
            popt, _ = curve_fit(double_gaussian, x_data, y_data, p0=initial_guess, maxfev=10000)
            
            # 使用拟合参数去噪
            y_fit = double_gaussian(x_data, *popt)
            y_denoised_gaussian = y_data - y_fit
            
            # 小波去噪
            wavelet = 'db1'
            coeffs = pywt.wavedec(y_denoised_gaussian, wavelet, level=6)
            coeffs_denoised = adaptive_thresholding(coeffs)
            
            # 信号重构
            y_denoised_wavelet = pywt.waverec(coeffs_denoised, wavelet)
            
            # 处理数据长度不一致问题
            min_length = min(len(x_data), len(y_denoised_wavelet))
            x_data = x_data[:min_length]
            y_data = y_data[:min_length]
            y_fit = y_fit[:min_length]
            y_denoised_wavelet = y_denoised_wavelet[:min_length]
            
            # 创建子图
            plt.figure(figsize=(15, 10))

            # 原始数据子图
            plt.subplot(3, 1, 1)
            plt.plot(x_data, y_data, label='Original Data')
            plt.legend()
            plt.title('Original Data')

            # 双高斯拟合子图
            plt.subplot(3, 1, 2)
            plt.plot(x_data, y_data, label='Original Data')
            plt.plot(x_data, y_fit, label='Fitted Gaussian', linestyle='dashed')
            plt.legend()
            plt.title('Gaussian Fitting')

            # 小波去噪子图
            plt.subplot(3, 1, 3)
            plt.plot(x_data, y_denoised_gaussian, label='Denoised Gaussian Data')
            plt.plot(x_data, y_denoised_wavelet, label='Wavelet Denoised Data', linestyle='dotted')
            plt.legend()
            plt.title('Wavelet Denoising')

            # 总体图像调整
            plt.tight_layout()
            plt.xlabel('Index')
            plt.ylabel('Value')

            # 保存图片
            output_image_path = os.path.join(output_folder, file_name.replace('.ini', '_combined.png'))
            plt.savefig(output_image_path)
            plt.close()
        
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
