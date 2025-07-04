# VMD 分解后得到的IMF和频谱图

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from vmdpy import VMD

# 设置字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 读取数据
df = pd.read_csv('DataSet/12k_1797_10c.csv', header=0, index_col=0)
data = df.iloc[:, :512].values
labels = df.iloc[:, 512].values

# 2. 取标签=9 的第一个窗口信号
signal = data[labels == 9][0]

# print(signal.shape)
# 3. VMD 分解参数
alpha = 2000     # 惩罚因子
tau = 0          # 时间步长
K = 4            # 模态数量
DC = 0
init = 1
tol = 1e-6

# 4. 执行 VMD
modes, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
fs = 12000  # 采样频率

# 5. 绘制 4 个 IMF 的时域图
plt.figure(figsize=(8, 6))
for i in range(K):
    plt.subplot(K, 1, i+1)
    plt.plot(np.arange(len(modes[i]))/fs, modes[i])
    plt.ylabel(f"IMF{i+1}")
    if i == K-1:
        plt.xlabel("时间 t/s")
plt.tight_layout()
plt.show()

# 6. 绘制 4 个 IMF 的频谱图
plt.figure(figsize=(8, 6))  # 增加图像尺寸
for i in range(K):
    f, Pxx = welch(modes[i], fs=fs, nperseg=256)
    plt.subplot(K, 1, i+1)
    plt.plot(f, Pxx)
    plt.ylabel(f"IMF{i+1}")
    if i == K-1:
        plt.xlabel("频率 f/Hz")
plt.suptitle("频谱分析", y=0.98)
plt.tight_layout()
plt.show()