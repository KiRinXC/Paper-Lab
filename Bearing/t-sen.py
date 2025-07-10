import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# === 加载特征与标签 ===
X_train = np.load('features_rcmwpe/X_train_features.npy')
X_test = np.load('features_rcmwpe/X_test_features.npy')
y_train = np.load('split_dataset/y_train.npy')
y_test = np.load('split_dataset/y_test.npy')

# === 合并用于可视化（也可以单独处理训练集）===
X_all = np.vstack((X_train, X_test))
y_all = np.concatenate((y_train, y_test))

# === 执行 t-SNE 降维 ===
print("正在执行 t-SNE 降维...")
tsne = TSNE(n_components=3, perplexity=30, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X_all)

# === 可视化 ===
label_names = ['Normal', 'IR007', 'B007', 'OR007']
colors = ['green', 'red', 'blue', 'orange']

plt.figure(figsize=(10, 6))
for label in np.unique(y_all):
    idx = y_all == label
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=label_names[label], s=20, alpha=0.7, c=colors[label])

plt.legend()
plt.title("t-SNE Visualization of RCMWPE Features")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.tight_layout()

# === 保存图像和降维结果 ===
save_dir = "tsne_results"
os.makedirs(save_dir, exist_ok=True)

plt.savefig(os.path.join(save_dir, 'tsne_plot.png'), dpi=300)
np.save(os.path.join(save_dir, 'X_tsne.npy'), X_tsne)
np.save(os.path.join(save_dir, 'y_tsne.npy'), y_all)

print("✅ t-SNE 降维结果已保存：")
print(f"- 降维数据：{save_dir}/X_tsne.npy")
print(f"- 标签数据：{save_dir}/y_tsne.npy")
print(f"- 可视化图：{save_dir}/tsne_plot.png")
