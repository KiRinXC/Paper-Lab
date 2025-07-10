import numpy as np
from sklearn.model_selection import train_test_split
import os

# === 加载样本和标签 ===
X = np.load('preprocessed_dataset/X_samples.npy')
y = np.load('preprocessed_dataset/y_labels.npy')

# === 划分训练集和测试集 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.6,  # 60% 测试集
    random_state=42,
    stratify=y      # 保证各类别比例一致
)

# === 输出信息 ===
print("训练集样本数：", X_train.shape[0])
print("测试集样本数：", X_test.shape[0])
print("训练集标签分布：", np.bincount(y_train))
print("测试集标签分布：", np.bincount(y_test))

# === 保存数据集 ===
save_dir = "split_dataset"
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
np.save(os.path.join(save_dir, 'y_test.npy'), y_test)

print(f"训练/测试集数据已保存至 {save_dir}/")

'''
训练集样本数： 160
测试集样本数： 240
训练集标签分布： [40 40 40 40]
测试集标签分布： [60 60 60 60]
训练/测试集数据已保存至 split_dataset/
'''

