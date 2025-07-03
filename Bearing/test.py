import numpy as np
import pandas as pd
from scipy.signal import welch
from vmdpy import VMD

# ---- 读取数据 ----
df = pd.read_csv('DataSet/12k_1797_10c.csv', header=0, index_col=0)
data = df.iloc[:, :512].values
labels = df.iloc[:, 512].values

# 取标签=9的所有窗口
idx9 = np.where(labels == 9)[0]
signals = data[idx9]

# ---- 中心频率计算 ----
def center_frequency(x, fs=12000, nperseg=256):
    f, Pxx = welch(x, fs=fs, nperseg=nperseg)
    return np.sum(f * Pxx) / np.sum(Pxx)

# ---- VMD 分解函数 ----
def vmd_modes(x, K, alpha=2000, tau=0, DC=0, init=1, tol=1e-6):
    """
    调用 vmdpy.VMD 返回 K 个模式分量 u
    参数说明：
    - alpha: 惩罚因子（平衡数据保真与稀疏性），这里可设 2000
    - tau: 双上升步长，一般 0
    - K: 模态数量
    - DC: 是否把第1模态当作直流，0/1
    - init: 频率初始化方式，1=均匀
    - tol: 终止阈值
    返回：u.shape = (K, N)
    """
    u, u_hat, omega = VMD(x, alpha, tau, K, DC, init, tol)
    return u

# ---- 对整个信号序列进行VMD分解 ----
K = 4
modes = vmd_modes(signals.flatten(), K)

# ---- 计算中心频率 ----
cfs = [center_frequency(modes[k]) for k in range(K)]

print("整个信号序列的中心频率 (Hz)：")
print(cfs)