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
# 4. 多尺度加权排列熵 (MWPE)
# ================================
def mwpe_vector(data, m=5, tau=1, max_scale=20):
    return [
        weighted_perm_entropy(coarse_grain(data, s), m, tau)
        for s in range(1, max_scale + 1)
    ]


# ================================
# 5. 计算每类 MWPE 特征
# ================================
def compute_class_mwpe_curves(X, y,
                              K=4, alpha=2000, tau_vmd=0., DC=0, init=1, tol=1e-7,
                              m=5, tau_mwpe=1, max_scale=20):
    n_classes = len(np.unique(y))
    curves = {cls: np.zeros(max_scale) for cls in range(n_classes)}
    counts = {cls: 0 for cls in range(n_classes)}

    for sample, label in zip(X, y):
        # VMD 分解信号
        imfs, _, _ = VMD(sample, alpha, tau_vmd, K, DC, init, tol)
        mwpe_sum = np.zeros(max_scale)

        # 对每个 IMF 计算 MWPE
        for k in range(K):
            mwpe_sum += np.array(mwpe_vector(imfs[k], m, tau_mwpe, max_scale))

        curves[label] += mwpe_sum / K
        counts[label] += 1

    # 计算每个类的平均特征
    for cls in curves:
        curves[cls] /= counts[cls]
    return curves


# ================================
# 主流程：使用VMD分解信号 + MWPE特征提取 + 保存特征
# ================================
if __name__ == '__main__':
    # 数据路径
    csv_path = '../DataSet/origin_sample_data.csv'

    # 获取训练、验证、测试集
    X_train, X_val, X_test, y_train, y_val, y_test = sliding_window_sampling(csv_path)

    # 计算 MWPE 特征（对所有类别）
    mwpe_curves = compute_class_mwpe_curves(X_train, y_train)

    # 将 MWPE 特征保存为 CSV 文件
    # 将 MWPE 特征与类别标签组合，创建 DataFrame
    feature_matrix = []
    labels = []

    for cls in mwpe_curves:
        feature_matrix.append(mwpe_curves[cls])
        labels.extend([cls] * len(mwpe_curves[cls]))

    # 转换为 numpy 数组并保存为 DataFrame
    # 将 MWPE 特征与类别标签组合，创建 DataFrame

    # ====== 将 MWPE 特征与类别标签组合，创建 DataFrame ======
    # feature_matrix: shape (n_classes, max_scale)
    feature_matrix = np.array([mwpe_curves[cls] for cls in mwpe_curves])
    # labels: shape (n_classes,)
    labels = np.array(sorted(mwpe_curves.keys()))

    # 创建 DataFrame
    # 每行对应一个类别，每列对应一个尺度上的特征
    df_features = pd.DataFrame(
        feature_matrix,
        columns=[f'Feature_{i + 1}' for i in range(feature_matrix.shape[1])]
    )
    df_features['Label'] = labels

    # 保存为 CSV
    df_features.to_csv('vmd_mwpe_features.csv', index=False)
    print("特征已经保存为 vmd_cmwpe_features.csv")

    # 打印提取的 MWPE 特征的形状
    print("提取的特征的形状：", feature_matrix.shape)


