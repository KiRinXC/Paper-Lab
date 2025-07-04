import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vmdpy import VMD
from scipy.fft import fft, fftfreq
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 设置采样频率
fs = 12000
signal_length = 4096
# 读取一段代表性信号（如 OR021）
signal = pd.read_csv('../DataSet/origin_sample_data.csv')['OR021'].values[:signal_length]

# VMD 参数设置
K = 4
alpha = 2000       # 惩罚因子
tau = 0.           # 拉格朗日乘子时间常数
DC = 0             # 不保留DC分量
init = 1           # 初始化方式
tol = 1e-7         # 收敛容差

# VMD 分解
u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)

# -------------------------------
# 🎨 时域图绘制
# -------------------------------
# 时间轴生成
time = np.arange(u.shape[1]) / fs  # 每个 IMF 都同样长度

fig1, axs = plt.subplots(K, 1, figsize=(10, 6))
for i in range(K):
    axs[i].plot(time, u[i], linewidth=1)
    axs[i].set_ylabel(f'IMF{i+1}')
    axs[i].grid(True)

axs[-1].set_xlabel('Time (s)')
axs[0].set_title(f'VMD 分解结果（K={K}）— 时域图')
plt.tight_layout()
plt.show()


# -------------------------------
# 🎨 频谱图绘制（幅度频率图）
# -------------------------------
N = u.shape[1]
freqs = fftfreq(N, d=1/fs)
fig2, axs2 = plt.subplots(K, 1, figsize=(10, 6))
for i in range(K):
    fft_vals = np.abs(fft(u[i]))[:N//2]
    axs2[i].plot(freqs[:N//2], fft_vals, linewidth=1)
    axs2[i].set_ylabel(f'IMF{i+1}')
    axs2[i].grid(True)
axs2[0].set_title(f'VMD 分解结果（K={K}）— 频谱图')
axs2[-1].set_xlabel('Frequency (Hz)')
plt.tight_layout()
plt.show()
