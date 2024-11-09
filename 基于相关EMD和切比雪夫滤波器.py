import os
import numpy as np
from scipy.signal import cheby1, filtfilt
from PyEMD import EMD
import matplotlib.pyplot as plt

def chebyshev_filter(signal, order=4, rp=1, Wn=0.1):
    b, a = cheby1(order, rp, Wn, btype='low', analog=False)
    if len(signal) > 15:
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    else:
        raise ValueError("The length of the input vector must be greater than 15.")

def apply_emd(signal):
    emd = EMD()
    imfs = emd.emd(signal)
    return imfs

def process_file(data_path, output_dir):
    try:
        data = np.loadtxt(data_path)
        if data.size == 0:
            raise ValueError(f"File {data_path} is empty.")
    except Exception as e:
        print(f"Error reading {data_path}: {e}")
        return
    
    try:
        filtered_data = chebyshev_filter(data, order=4, rp=1, Wn=0.1)
    except Exception as e:
        print(f"Error filtering data from {data_path}: {e}")
        return
    
    imfs = apply_emd(filtered_data)
    reconstructed_signal = np.sum(imfs, axis=0)
    
    print(f"处理文件: {data_path}")
    
    filename = os.path.splitext(os.path.basename(data_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(data, label='Original Data')
    plt.title('Original Data')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{filename}_original_data.png'))
    
    plt.subplot(3, 1, 2)
    plt.plot(filtered_data, label='Filtered Data (Chebyshev)')
    plt.title('Filtered Data (Chebyshev)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{filename}_filtered_data.png'))
    
    plt.subplot(3, 1, 3)
    plt.plot(reconstructed_signal, label='Reconstructed Signal (EMD)')
    plt.title('Reconstructed Signal (EMD)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{filename}_reconstructed_signal.png'))
    
    plt.close()
    
    plt.figure(figsize=(12, 8))
    for i, imf in enumerate(imfs):
        plt.subplot(len(imfs), 1, i+1)
        plt.plot(imf, label=f'IMF {i+1}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{filename}_imf_{i+1}.png'))
    plt.close()

def main(data_dir):
    output_dir = os.path.join(data_dir, 'output')
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.ini'):
            data_path = os.path.join(data_dir, filename)
            file_output_dir = os.path.join(output_dir, filename)
            os.makedirs(file_output_dir, exist_ok=True)
            process_file(data_path, file_output_dir)

if __name__ == "__main__":
    data_dir = 'D:\shudeng\ProofingTool\数据'
    main(data_dir)
