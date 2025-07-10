import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vmdpy import VMD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
import math
from antropy import perm_entropy

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False


# ================================
# 滑动窗口采样 + 划分 + 标准化
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
# 复合多尺度加权排列熵 (CMWPE)
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


def coarse_grain(signal, scale):
    length = len(signal)
    M = length // scale
    return np.array([np.mean(signal[i * scale:(i + 1) * scale]) for i in range(M)])


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


def compute_class_cmwpe_curves(X, y,
                               K=4, alpha=2000, tau_vmd=0., DC=0, init=1, tol=1e-7,
                               m=5, tau_cmwpe=1, max_scale=20):
    n_classes = len(np.unique(y))
    curves = {cls: np.zeros(max_scale) for cls in range(n_classes)}
    counts = {cls: 0 for cls in range(n_classes)}

    for sample, label in zip(X, y):
        imfs, _, _ = VMD(sample, alpha, tau_vmd, K, DC, init, tol)
        cmwpe_sum = np.zeros(max_scale)
        for k in range(K):
            cmwpe_sum += np.array(composite_mwpe(imfs[k], m=m, tau=tau_cmwpe, max_scale=max_scale))
        curves[label] += cmwpe_sum / K
        counts[label] += 1

    for cls in curves:
        curves[cls] /= counts[cls]
    return curves


# ================================
# 多尺度排列熵 (MPE)
# ================================
def mpe_vector(data, m=5, tau=1, max_scale=20):
    return [
        perm_entropy(coarse_grain(data, s), order=m, delay=tau, normalize=True)
        for s in range(1, max_scale + 1)
    ]


def compute_class_mpe_curves(X, y,
                             K=4, alpha=2000, tau_vmd=0., DC=0, init=1, tol=1e-7,
                             m=5, tau_mpe=1, max_scale=20):
    n_classes = len(np.unique(y))
    curves = {cls: np.zeros(max_scale) for cls in range(n_classes)}
    counts = {cls: 0 for cls in range(n_classes)}

    for sample, label in zip(X, y):
        imfs, _, _ = VMD(sample, alpha, tau_vmd, K, DC, init, tol)
        mpe_sum = np.zeros(max_scale)
        for k in range(K):
            mpe_sum += np.array(mpe_vector(imfs[k], m, tau_mpe, max_scale))
        curves[label] += mpe_sum / K
        counts[label] += 1

    for cls in curves:
        curves[cls] /= counts[cls]
    return curves


# ================================
# 多尺度加权排列熵 (MWPE)
# ================================
def weighted_perm_entropy(data, m=5, tau=1):
    N = len(data)
    patterns = []
    weights = []
    for i in range(N - (m - 1) * tau):
        seq = data[i:i + (m - 1) * tau + 1:tau]
        rank = tuple(np.argsort(seq))
        patterns.append(rank)
        weights.append(np.sum(np.abs(seq)))

    total_weight = np.sum(weights)
    prob = {}
    for p, w in zip(patterns, weights):
        prob[p] = prob.get(p, 0) + w

    entropy = -sum((w / total_weight) * np.log(w / total_weight + 1e-12) for w in prob.values())
    max_entropy = np.log(math.factorial(m))
    return entropy / max_entropy


def mwpe_vector(data, m=5, tau=1, max_scale=20):
    return [
        weighted_perm_entropy(coarse_grain(data, s), m, tau)
        for s in range(1, max_scale + 1)
    ]


def compute_class_mwpe_curves(X, y,
                              K=4, alpha=2000, tau_vmd=0., DC=0, init=1, tol=1e-7,
                              m=5, tau_mwpe=1, max_scale=20):
    n_classes = len(np.unique(y))
    curves = {cls: np.zeros(max_scale) for cls in range(n_classes)}
    counts = {cls: 0 for cls in range(n_classes)}

    for sample, label in zip(X, y):
        imfs, _, _ = VMD(sample, alpha, tau_vmd, K, DC, init, tol)
        mwpe_sum = np.zeros(max_scale)
        for k in range(K):
            mwpe_sum += np.array(mwpe_vector(imfs[k], m, tau_mwpe, max_scale))
        curves[label] += mwpe_sum / K
        counts[label] += 1

    for cls in curves:
        curves[cls] /= counts[cls]
    return curves


# ================================
# 主流程 + 绘图
# ================================
if __name__ == '__main__':
    csv_path = '../DataSet/origin_sample_data.csv'
    X_train, X_val, X_test, y_train, y_val, y_test = sliding_window_sampling(csv_path)

    # 筛选直径0.007对应的类别
    target = [0, 1, 2, 3]  # normal, IR007, B007, OR007
    mask = np.isin(y_train, target)
    X_f, y_f = X_train[mask], y_train[mask]

    # 计算 CMWPE 曲线
    cmwpe_curves = compute_class_cmwpe_curves(X_f, y_f)

    # 计算 MPE 曲线
    mpe_curves = compute_class_mpe_curves(X_f, y_f)

    # 计算 MWPE 曲线
    mwpe_curves = compute_class_mwpe_curves(X_f, y_f)

    # 绘图
    scales = np.arange(1, 21)
    labels = ['normal', 'IR007', 'B007', 'OR007']

    # CMWPE 绘图
    plt.figure(figsize=(8, 6))
    for cls in target:
        plt.plot(scales, cmwpe_curves[cls], marker='o', label=f'CMWPE-{labels[cls]}')

    plt.xlabel('尺度因子 $s$', fontsize=12)
    plt.ylabel('CMWPE 熵值', fontsize=12)
    plt.title('复合多尺度加权排列熵（CMWPE）曲线', fontsize=14)
    plt.xticks(np.arange(0, 21, 1))  # 横坐标0, 1, 2, ..., 20
    plt.yticks(np.arange(0.5, 0.9, 0.05))
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # MPE 绘图
    plt.figure(figsize=(8, 6))
    for cls in target:
        plt.plot(scales, mpe_curves[cls], marker='x', label=f'MPE-{labels[cls]}')

    plt.xlabel('尺度因子 $s$', fontsize=12)
    plt.ylabel('MPE 熵值', fontsize=12)
    plt.title('多尺度排列熵（MPE）曲线', fontsize=14)
    plt.xticks(np.arange(0, 21, 1))  # 横坐标0, 1, 2, ..., 20
    plt.yticks(np.arange(0.5, 0.9, 0.05))
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # MWPE 绘图
    plt.figure(figsize=(8, 6))
    for cls in target:
        plt.plot(scales, mwpe_curves[cls], marker='^', label=f'MWPE-{labels[cls]}')

    plt.xlabel('尺度因子 $s$', fontsize=12)
    plt.ylabel('MWPE 熵值', fontsize=12)
    plt.title('多尺度加权排列熵（MWPE）曲线', fontsize=14)
    plt.xticks(np.arange(0, 21, 1))  # 横坐标0, 1, 2, ..., 20
    plt.yticks(np.arange(0.5, 0.9, 0.05))
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
