embedding_dim = 5   # m
delay = 1           # τ
scales = range(1, 21)  # s: 1 到 20

import numpy as np
import os


def compute_weighted_entropy(series, m, tau):
    N = len(series)
    embedded = np.array([series[i:i + m * tau:tau] for i in range(N - (m - 1) * tau)])
    patterns = np.argsort(embedded, axis=1)

    # 权重计算（考虑幅值信息）
    means = np.mean(embedded, axis=1, keepdims=True)
    weights = np.mean((embedded - means) ** 2, axis=1)

    # 构建模式统计
    unique_patterns, inverse, counts = np.unique(patterns, axis=0, return_inverse=True, return_counts=True)
    weighted_counts = np.zeros(len(unique_patterns))

    for idx, count in enumerate(inverse):
        weighted_counts[count] += weights[idx]

    probs = weighted_counts / np.sum(weighted_counts)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return entropy


def compute_rcmwpe(signal, m=5, tau=1, scales=range(1, 21)):
    rcmwpe = []
    N = len(signal)
    for s in scales:
        pe_all = []
        for j in range(s):
            # 精细复合粗粒化
            coarse = [(np.mean(signal[j + i * s: j + i * s + s]) if (j + i * s + s) <= N else 0)
                      for i in range((N - j) // s)]
            if len(coarse) < m:
                pe_all.append(0)
            else:
                pe_all.append(compute_weighted_entropy(np.array(coarse), m, tau))
        rcmwpe.append(np.mean(pe_all))
    return np.array(rcmwpe)  # shape: (20,)


def extract_features_from_vmd_set(X_vmd, m=5, tau=1, scales=range(1, 21)):
    all_features = []
    for sample in X_vmd:  # shape: (4, 2048)
        sample_feats = []
        for imf in sample:
            feat = compute_rcmwpe(imf, m, tau, scales)
            sample_feats.append(feat)
        all_features.append(np.concatenate(sample_feats))  # shape: (4*20,)
    return np.array(all_features)  # shape: [num_samples, 80]

# === 加载 VMD 分解结果 ===
X_train_vmd = np.load('vmd_decomposed/X_train_vmd.npy')
X_test_vmd = np.load('vmd_decomposed/X_test_vmd.npy')

# === 特征提取 ===
print("正在提取训练集 RCMWPE 特征...")
X_train_features = extract_features_from_vmd_set(X_train_vmd)

print("正在提取测试集 RCMWPE 特征...")
X_test_features = extract_features_from_vmd_set(X_test_vmd)

# === 保存 ===
save_dir = "features_rcmwpe"
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, 'X_train_features.npy'), X_train_features)
np.save(os.path.join(save_dir, 'X_test_features.npy'), X_test_features)

print("训练集特征 shape:", X_train_features.shape)  # [样本数, 80]
print("测试集特征 shape:", X_test_features.shape)
