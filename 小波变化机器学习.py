import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from deap import base, creator, tools, algorithms

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

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

# 基于贝叶斯推断的自适应阈值函数，确保返回标量
def bayesian_threshold(wavelet_coeffs):
    sigma = np.std(wavelet_coeffs)
    threshold = sigma * np.sqrt(2 * np.log(len(wavelet_coeffs)))
    return threshold

# 自适应小波阈值函数，确保阈值为标量
def adaptive_wavelet_threshold(wavelet_coeffs, threshold, alpha):
    # 确保 threshold 是标量
    if np.isscalar(threshold):
        threshold = float(threshold)
    result_coeffs = np.sign(wavelet_coeffs) * np.maximum(np.abs(wavelet_coeffs) - threshold, 0) ** alpha
    return result_coeffs

# 自适应小波去噪函数，结合自适应阈值
def adaptive_wavelet_denoising(signal, wavelet_name='coif5', alpha=1.0, level=5):
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    thresholded_coeffs = []
    
    for i in range(len(coeffs)):
        threshold = bayesian_threshold(coeffs[i])  # 使用贝叶斯阈值
        thresholded_coeffs.append(adaptive_wavelet_threshold(coeffs[i], threshold, alpha))
    
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet_name)
    
    # 修正长度以匹配原始信号
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    else:
        denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), mode='constant')
    
    return denoised_signal

# 交叉验证评分函数
def cross_validation_score(signal, original_signal, wavelet_name, alpha, level, n_splits=5):
    kf = KFold(n_splits=n_splits)
    snr_scores = []
    
    for train_index, test_index in kf.split(signal):
        train_signal, test_signal = signal[train_index], signal[test_index]
        denoised_train_signal = adaptive_wavelet_denoising(train_signal, str(wavelet_name), alpha, level)
        snr = calculate_snr(test_signal, denoised_train_signal[:len(test_signal)])
        snr_scores.append(snr)
    
    return np.mean(snr_scores)

# 自定义交叉操作：保留小波基，交叉alpha和level
def custom_crossover(ind1, ind2):
    ind1[1], ind2[1] = tools.cxBlend(ind1[1:], ind2[1:], alpha=0.5)
    return ind1, ind2

# 动态选择小波基 + 遗传算法进行自适应优化
def optimize_parameters(signal, original_signal, wavelet_list, alpha_range=(0.1, 2.0), level_range=(2, 7), n_gen=20, pop_size=30):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_wavelet", np.random.choice, [str(w) for w in wavelet_list])  # 确保所有小波基为字符串
    toolbox.register("attr_alpha", np.random.uniform, alpha_range[0], alpha_range[1])
    toolbox.register("attr_level", np.random.randint, level_range[0], level_range[1] + 1)
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     (toolbox.attr_wavelet, toolbox.attr_alpha, toolbox.attr_level), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        wavelet, alpha, level = individual
        snr_score = cross_validation_score(signal, original_signal, str(wavelet), alpha, level)
        return snr_score,

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", custom_crossover)  # 使用自定义交叉操作
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen, verbose=True)

    best_individual = tools.selBest(pop, k=1)[0]
    best_wavelet, best_alpha, best_level = best_individual
    best_snr = evaluate(best_individual)[0]

    return best_wavelet, best_alpha, best_level, best_snr

# 可视化信号与去噪效果，并保存图像
def plot_signal_comparison(original_signal, denoised_signal, file_name, output_folder):
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(original_signal)
    plt.title(f'原始信号 - {file_name}')

    plt.subplot(2, 1, 2)
    plt.plot(denoised_signal)
    plt.title(f'自适应小波阈值去噪信号 - {file_name}')

    plt.tight_layout()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_path = os.path.join(output_folder, f'{file_name}.png')
    plt.savefig(output_path)
    plt.close()

# 处理文件夹中所有文件，并自动优化小波基、alpha和层数
def process_folder(input_folder, output_folder):
    wavelet_list = pywt.wavelist(kind='discrete')  # 获取所有小波基列表
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".ini"):
            file_path = os.path.join(input_folder, file_name)
            signal = load_data_from_ini(file_path)
            if len(signal) == 0:
                print(f"文件 {file_name} 为空，跳过处理。")
                continue
            
            # 将原始信号作为基准（没有去噪之前的信号）
            original_signal = signal.copy()
            
            # 优化寻找最佳参数
            best_wavelet, best_alpha, best_level, best_snr = optimize_parameters(
                signal, original_signal, wavelet_list)
            print(f"文件 {file_name} 的最佳小波基为: {best_wavelet}, 最佳 alpha 为: {best_alpha}, 最佳层数为: {best_level}, SNR: {best_snr}")
            
            # 使用最优参数进行去噪
            denoised_signal = adaptive_wavelet_denoising(signal, wavelet_name=str(best_wavelet), alpha=best_alpha, level=best_level)
            
            # 保存去噪后的结果图像
            plot_signal_comparison(original_signal, denoised_signal, file_name.split('.')[0], output_folder)

# 主程序入口
if __name__ == "__main__":
    input_folder = r"D:\shudeng\boxongtu\log"
    output_folder = r"D:\shudeng\boxongtu\不同小波基小波预处理结果chushiban"
    process_folder(input_folder, output_folder)
