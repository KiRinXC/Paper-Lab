import numpy as np
import pandas as pd
from scipy.signal import welch
from vmdpy import VMD

# ---- 读取数据 ----
df     = pd.read_csv('DataSet/12k_1797_10c.csv', header=0, index_col=0)
data   = df.iloc[:, :512].values
labels = df.iloc[:, 512].values

# 取标签=9的第一个窗口
# idx9   = np.where(labels == 9)[0]
# signal = data[idx9[0]]

idx9 = np.where(labels == 9)[0]
signal = data[idx9]

# ---- 中心频率计算 ----
def center_frequency(x, fs=12000, nperseg=512):
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

# ---- 批量分解并计算中心频率 ----
Ks    = [2, 3, 4, 5,6]
table = {}

for K in Ks:
    modes = vmd_modes(signal, K)
    # 每行一个模态，计算其中心频率
    cfs = [center_frequency(modes[k]) for k in range(K)]
    table[K] = cfs

# ---- 构造并打印表格 ----
max_k = max(Ks)
df_cf = pd.DataFrame.from_dict(
    {K: (table[K] + [np.nan]*(max_k-K)) for K in Ks},
    orient='index',
    columns=[f"IMF{i+1}" for i in range(max_k)]
).round(0).astype('Int64')
df_cf.index.name = 'K'

print("VMD 分解 —— 不同 K 对应的中心频率 (Hz)：")
print(df_cf)


'''
VMD 分解 —— 不同 K 对应的中心频率 (Hz)：
   IMF1  IMF2  IMF3  IMF4  IMF5  IMF6
K                                    
2   563  2791  <NA>  <NA>  <NA>  <NA>
3   562  2769  3451  <NA>  <NA>  <NA>
4   560  1426  2769  3451  <NA>  <NA>
5   559  1420  2767  3394  3590  <NA>
6   559  1399  2640  2815  3398  3595
'''