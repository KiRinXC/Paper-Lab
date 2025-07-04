import numpy as np
import pandas as pd
from vmdpy import VMD
from scipy.fft import fft, fftfreq

def calc_center_frequency(signal, fs=12000):
    # 计算 FFT 并提取频谱能量
    N = len(signal)
    freqs = fftfreq(N, d=1/fs)
    fft_vals = np.abs(fft(signal))[:N//2]
    freqs = freqs[:N//2]
    center_freq = np.sum(freqs * fft_vals) / np.sum(fft_vals)
    return center_freq

def vmd_center_freqs(signal, k_list, alpha=2000, tau=0., DC=0, init=1, tol=1e-7, fs=12000):
    result_dict = {}

    for K in k_list:
        u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
        cf_list = [calc_center_frequency(imf, fs) for imf in u]
        result_dict[K] = cf_list
        print(f"K = {K}, 中心频率: {np.round(cf_list, 2)} Hz")

    return result_dict

signal_length = 4096
# 读取 OR021 的原始信号（未归一化的 CSV）
signal = pd.read_csv('../DataSet/origin_sample_data.csv')['OR021'].values[:signal_length]

# 尝试的 K 值范围
K_range = list(range(2, 7))

# 执行分析
center_freqs = vmd_center_freqs(signal, K_range)
'''
K = 2, 中心频率: [ 776.17 2908.68] Hz
K = 3, 中心频率: [ 713.04 2785.72 3469.8 ] Hz
K = 4, 中心频率: [ 674.36 1520.84 2798.28 3475.54] Hz
K = 5, 中心频率: [ 660.83 1490.57 2773.3  3364.54 3570.79] Hz
K = 6, 中心频率: [ 649.04 1455.36 2649.66 2857.17 3385.11 3585.55] Hz
'''