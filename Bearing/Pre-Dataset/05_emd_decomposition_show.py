import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import EMD
from scipy.fft import fft, fftfreq
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 设置参数
fs = 12000  # 采样频率
signal_length = 4096

# 读取信号（同样用 OR021）
signal = pd.read_csv('../DataSet/origin_sample_data.csv')['OR021'].values[:signal_length]

# 执行 EMD 分解
emd = EMD()
imfs = emd(signal)

# IMF 数量
num_imfs = imfs.shape[0]
print(f"得到 {num_imfs} 个 IMF 分量")

# -------------------------------
# 🎨 时域图绘制（含时间轴单位）
# -------------------------------
time = np.arange(signal_length) / fs
fig1, axs = plt.subplots(num_imfs, 1, figsize=(10, 1.5 * num_imfs))
for i in range(num_imfs):
    axs[i].plot(time, imfs[i], linewidth=1)
    axs[i].set_ylabel(f'IMF{i+1}')
    axs[i].grid(True)
axs[-1].set_xlabel('Time (s)')
axs[0].set_title('EMD 分解结果 - 时域图')
plt.tight_layout()
plt.show()

# -------------------------------
# 🎨 频谱图绘制（幅度 vs 频率）
# -------------------------------
freqs = fftfreq(signal_length, d=1/fs)
fig2, axs2 = plt.subplots(num_imfs, 1, figsize=(10, 1.5 * num_imfs))
for i in range(num_imfs):
    fft_vals = np.abs(fft(imfs[i]))[:signal_length // 2]
    axs2[i].plot(freqs[:signal_length // 2], fft_vals, linewidth=1)
    axs2[i].set_ylabel(f'IMF{i+1}')
    axs2[i].grid(True)
axs2[-1].set_xlabel('Frequency (Hz)')
axs2[0].set_title('EMD 分解结果 - 频谱图')
plt.tight_layout()
plt.show()
