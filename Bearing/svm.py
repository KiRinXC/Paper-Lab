import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 固定随机种子
np.random.seed(42)

# 加载数据
X = np.load('preprocessed_dataset/X_samples.npy')
y = np.load('preprocessed_dataset/y_labels.npy')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.6, random_state=42, stratify=y)

# 初始化 SVM 模型（不使用优化算法）
# 你可以手动修改 C 和 gamma 的值
model = SVC(C=1.0, gamma=0.1, kernel='rbf')  # 示例参数

# 训练模型
model.fit(X_train, y_train)

# 测试集预测
y_pred = model.predict(X_test)

# 准确率
acc = accuracy_score(y_test, y_pred)
print("✅ 测试集准确率：", round(acc * 100,2), "%")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
labels = ['Normal', 'IR014', 'B014', 'OR014']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# 可视化
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax, colorbar=True, values_format='d')
plt.title("Standard SVM Confusion Matrix on Test Set")
plt.tight_layout()
plt.savefig("svm_confusion_matrix.png", dpi=300)
plt.show()
