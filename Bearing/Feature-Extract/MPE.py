import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from vmdpy import VMD
from antropy import perm_entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# ================================
# 1. 滑动窗口采样 + 划分 + 标准化
# ================================
def sliding_window_sampling(csv_path, win_size=2048, overlap_ratio=0.5,
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
# 2. 多尺度排列熵工具函数
# ================================
def coarse_grain(data, scale):
    N = len(data)
    M = N // scale
    return np.array([data[i * scale:(i + 1) * scale].mean() for i in range(M)])


def mpe_vector(data, m=5, tau=1, max_scale=20):
    """返回尺度 1…max_scale 下的 MPE 特征列表"""
    return [
        perm_entropy(coarse_grain(data, s), order=m, delay=tau, normalize=True)
        for s in range(1, max_scale + 1)
    ]


# ================================
# 3. 计算每类 MPE 曲线
# ================================
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

    # 平均化
    for cls in curves:
        curves[cls] /= counts[cls]

    return curves


# ================================
# 4. 主流程 + 绘图
# ================================
if __name__ == '__main__':
    # A. 滑窗采样与划分
    csv_path = '../DataSet/origin_sample_data.csv'
    X_train, X_val, X_test, y_train, y_val, y_test = sliding_window_sampling(csv_path)

    # B. 只保留目标类别的数据
    target_classes = [0, 1, 2, 3]  # normal, IR007, B007, OR007
    train_filter = np.isin(y_train, target_classes)
    val_filter = np.isin(y_val, target_classes)
    test_filter = np.isin(y_test, target_classes)

    # 筛选出目标类别的数据
    X_train_filtered = X_train[train_filter]
    y_train_filtered = y_train[train_filter]
    X_val_filtered = X_val[val_filter]
    y_val_filtered = y_val[val_filter]
    X_test_filtered = X_test[test_filter]
    y_test_filtered = y_test[test_filter]

    # C. 计算训练集各类 MPE 曲线
    curves = compute_class_mpe_curves(X_train_filtered, y_train_filtered)

    # D. 绘制多尺度 MPE 曲线对比（仅 4 条）
    scales = np.arange(1, 21)
    labels = ['normal', 'IR007', 'B007', 'OR007']

    # 只绘制这四个类别
    plot_classes = [0, 1, 2, 3]  # normal, IR007, B007, OR007

    plt.figure(figsize=(8, 6))
    for cls in plot_classes:
        plt.plot(scales, curves[cls], marker='o', label=labels[cls])

    # 设置x轴刻度为0, 2, 4, 6, 8...
    plt.xticks(np.arange(0, 21, 2))

    plt.xlabel('尺度因子 $s$', fontsize=12)
    plt.ylabel('MPE', fontsize=12)
    plt.title('直径0.007″下各类 VMD(IMF平均) 多尺度排列熵曲线', fontsize=14)
    plt.xticks(np.arange(0, 21, 2))  # 横坐标0, 2, 4, ..., 20
    plt.yticks(np.arange(0.4, 1.1, 0.1))  # 纵坐标0.4到1.0
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
