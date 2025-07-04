import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sliding_window_sampling(csv_path, win_size=512, overlap_ratio=0.5,
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

if __name__ == '__main__':
    csv_path = '../DataSet/origin_sample_data.csv'
    X_train, X_val, X_test, y_train, y_val, y_test = sliding_window_sampling(csv_path)

    print("训练集样本数：", len(y_train))
    print("验证集样本数：", len(y_val))
    print("测试集样本数：", len(y_test))
