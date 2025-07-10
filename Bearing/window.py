import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os

# === 参数设置 ===
win_size = 2048   # 滑动窗口长度
step = 1024       # 滑动步长
target_labels = ['normal', 'IR007', 'B007', 'OR007']
label_map = {label: idx for idx, label in enumerate(target_labels)}

# === 加载数据 ===
file_path = './Dataset/origin_sample_data.csv'
df = pd.read_csv(file_path)
df = df[target_labels]

# === 滑动窗口处理函数 ===
def sliding_window(data_array, label_idx, win_size=2048, step=1024):
    samples = []
    labels = []
    for i in range(0, len(data_array) - win_size + 1, step):
        window = data_array[i:i + win_size]
        samples.append(window)
        labels.append(label_idx)
    return np.array(samples), np.array(labels)

# === 构建所有样本和标签 ===
X_all, y_all = [], []
for label in target_labels:
    signal = df[label].values
    label_idx = label_map[label]
    X, y = sliding_window(signal, label_idx, win_size=win_size, step=step)
    X_all.append(X)
    y_all.append(y)

# === 合并并打乱 ===
X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)
X_all, y_all = shuffle(X_all, y_all, random_state=42)

print("样本集 shape：", X_all.shape)  # [样本数, 2048]
print("标签分布：", np.bincount(y_all))  # 每类样本数量

# === 保存为 .npy 文件 ===
save_dir = "preprocessed_dataset"
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, 'X_samples.npy'), X_all)
np.save(os.path.join(save_dir, 'y_labels.npy'), y_all)

print(f"数据已保存到 {save_dir}/ 目录下。")
