import numpy as np
import pandas as pd
from vmdpy import VMD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

# ================================
# 1. 滑动窗口采样 + 划分 + 标准化
# ================================
def sliding_window_sampling(csv_path, win_size=1024, overlap_ratio=0.5,
                            test_size=0.2, val_size=0.2, random_state=42):
    stride = int(win_size * (1 - overlap_ratio))
    data = pd.read_csv(csv_path)

    X_all = []
    y_all = []

    # 遍历每一列（即每一类信号）
    for label, (col_name, signal) in enumerate(data.items()):
        signal = np.array(signal)
        max_start = len(signal) - win_size + 1

        for start in range(0, max_start, stride):
            window = signal[start:start + win_size]
            X_all.append(window)
            y_all.append(label)

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    print(f"总样本数：{X_all.shape[0]}，每个样本长度：{X_all.shape[1]}，标签种类数：{np.unique(y_all).size}")

    # 第一步：划分训练+验证集与测试集
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all, y_all, test_size=test_size, stratify=y_all, random_state=random_state)

    # 第二步：再从训练+验证集中划分出验证集
    val_ratio_adjusted = val_size / (1 - test_size)  # 保证整体比例不变
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio_adjusted, stratify=y_trainval, random_state=random_state)

    # 第三步：Z-score 标准化（仅用训练集统计量）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


# ================================
# 2. 单尺度加权排列熵 (WPE)
# ================================
def weighted_perm_entropy(data, m=5, tau=1):
    N = len(data)
    patterns, weights = [], []
    for i in range(N - (m - 1) * tau):
        seq = data[i: i + (m - 1) * tau + 1: tau]
        patterns.append(tuple(np.argsort(seq)))
        weights.append(np.sum(np.abs(seq)))
    total_weight = np.sum(weights)
    prob = {}
    for p, w in zip(patterns, weights):
        prob[p] = prob.get(p, 0) + w
    p_vals = np.array(list(prob.values())) / (total_weight + 1e-12)
    H = -np.sum(p_vals * np.log(p_vals + 1e-12))
    return H / np.log(math.factorial(m))


# ================================
# 3. 粗粒化函数
# ================================
def coarse_grain(signal, scale):
    length = len(signal)
    M = length // scale
    return np.array([np.mean(signal[i * scale:(i + 1) * scale]) for i in range(M)])


# ================================
# 4. 复合多尺度加权排列熵 (CMWPE)
# ================================
def composite_mwpe(signal, m=5, tau=1, max_scale=20):
    cmwpe_vals = []
    for s in range(1, max_scale + 1):
        ent_sum, count = 0.0, 0
        for offset in range(s):
            truncated = signal[offset:]
            M = len(truncated) // s
            if M <= (m - 1) * tau:
                continue
            cg = coarse_grain(truncated, s)
            ent_sum += weighted_perm_entropy(cg, m=m, tau=tau)
            count += 1
        cmwpe_vals.append(ent_sum / count if count > 0 else 0.0)
    return cmwpe_vals


# ================================
# 5. 计算每类 CMWPE 特征
# ================================
def compute_class_cmwpe_curves(X, y,
                               K=4, alpha=2000, tau_vmd=0., DC=0, init=1, tol=1e-7,
                               m=5, tau_cmwpe=1, max_scale=20):
    n_samples = X.shape[0]
    curves = []
    for i in range(n_samples):
        sample = X[i]
        imfs, _, _ = VMD(sample, alpha, tau_vmd, K, DC, init, tol)
        cmwpe_sum = np.zeros(max_scale)

        for k in range(K):
            cmwpe_sum += np.array(composite_mwpe(imfs[k], m, tau_cmwpe, max_scale))

        curves.append(cmwpe_sum / K)

    # 每个样本的特征，曲线（cmwpe_sum/K）为每个窗口的特征
    return np.array(curves), y


# ================================
# 主流程：使用VMD分解信号 + CMWPE特征提取 + 保存特征
# ================================
if __name__ == '__main__':
    # 数据路径
    csv_path = '../DataSet/origin_sample_data.csv'

    # 获取训练、验证、测试集
    X_train, X_val, X_test, y_train, y_val, y_test = sliding_window_sampling(csv_path)

    # 计算 CMWPE 特征（对所有窗口）
    cmwpe_curves, labels = compute_class_cmwpe_curves(X_train, y_train)

    # 将 CMWPE 特征与标签组合，创建 DataFrame
    df_features = pd.DataFrame(cmwpe_curves, columns=[f'Feature_{i + 1}' for i in range(cmwpe_curves.shape[1])])
    df_features['Label'] = labels

    # 保存为 CSV 文件
    df_features.to_csv('vmd_cmwpe_features.csv', index=False)
    print("特征已经保存为 vmd_cmwpe_features.csv")

    # 打印提取的 CMWPE 特征的形状
    print("提取的特征的形状：", cmwpe_curves.shape)
