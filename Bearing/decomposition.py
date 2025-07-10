import numpy as np
from vmdpy import VMD
import os

# VMD 参数
alpha = 2000       # 惩罚因子
tau = 0            # noise-tolerance (0=no strict fidelity)
K = 4              # 模态数
DC = 0             # no DC part imposed
init = 1           # initialize omegas uniformly
tol = 1e-7         # convergence tolerance

def apply_vmd_to_dataset(X_set):
    """
    对每个样本执行 VMD 分解，返回形状为 [样本数, K, win_size]
    """
    vmd_features = []
    for signal in X_set:
        u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
        vmd_features.append(u)
    return np.array(vmd_features)  # shape: [num_samples, K, 2048]

# === 加载数据 ===
X_train = np.load('split_dataset/X_train.npy')
X_test = np.load('split_dataset/X_test.npy')

# === 对训练集/测试集执行 VMD 分解 ===
print("正在对训练集进行 VMD 分解...")
X_train_vmd = apply_vmd_to_dataset(X_train)

print("正在对测试集进行 VMD 分解...")
X_test_vmd = apply_vmd_to_dataset(X_test)

# === 保存分解后的数据 ===
save_dir = "vmd_decomposed"
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, 'X_train_vmd.npy'), X_train_vmd)
np.save(os.path.join(save_dir, 'X_test_vmd.npy'), X_test_vmd)

print(f"VMD 分解结果已保存至 {save_dir}/ 目录")
print("训练集 VMD shape:", X_train_vmd.shape)  # [num_samples, 4, 2048]
