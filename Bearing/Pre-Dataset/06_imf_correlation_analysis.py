import numpy as np
import pandas as pd
from PyEMD import EMD, EEMD, CEEMDAN
from vmdpy import VMD
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

# 设置参数
fs = 12000
signal_length = 4096

# 读取信号（OR021）
signal = pd.read_csv('../DataSet/origin_sample_data.csv')['OR021'].values[:signal_length]

# 定义互相关函数（皮尔逊相关系数）
def compute_correlations(imfs, original, num_imf=4):
    correlations = []
    for i in range(min(num_imf, imfs.shape[0])):
        corr, _ = pearsonr(imfs[i], original)
        correlations.append(corr)
    # 若 IMF 少于4个，用 NaN 填补
    while len(correlations) < num_imf:
        correlations.append(np.nan)
    return correlations

# 存放结果的 DataFrame
results = pd.DataFrame(index=[f'IMF{i+1}' for i in range(4)],
                       columns=['EMD', 'EEMD', 'CEEMDAN', 'VMD'])

# EMD 分解
emd = EMD()
emd_imfs = emd(signal)
results['EMD'] = compute_correlations(emd_imfs, signal)

# EEMD 分解
eemd = EEMD()
eemd_imfs = eemd(signal)
results['EEMD'] = compute_correlations(eemd_imfs, signal)

# CEEMDAN 分解
ceemdan = CEEMDAN()
ceemdan_imfs = ceemdan(signal)
results['CEEMDAN'] = compute_correlations(ceemdan_imfs, signal)

# VMD 分解（K=4）
K = 4
alpha = 2000
tau = 0
DC = 0
init = 1
tol = 1e-7
vmd_imfs, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
results['VMD'] = compute_correlations(vmd_imfs, signal)

# 转置 DataFrame
# results = results.transpose()

# 显示结果
print("\n互相关系数比较表（与原始信号）：")
print(results.round(4))