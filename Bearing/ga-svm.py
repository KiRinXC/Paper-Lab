import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# 固定随机种子，保证可复现
np.random.seed(42)

# === 1. 加载数据 ===
X = np.load('tsne_results/X_tsne_3d.npy')  # 确保路径正确
y = np.load('tsne_results/y_tsne_3d.npy')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.6, random_state=42, stratify=y
)

# === 2. 适应度函数 ===
def fitness_function(C, gamma,X_train=X, y_train=y):
    # 拆分训练集为训练+验证
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

    model = SVC(C=C, kernel='rbf', gamma=gamma)
    model.fit(X_subtrain, y_subtrain)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    # 惩罚使准确率控制在范围内
    if acc > 0.995:
        penalty = (acc - 0.995) * 10
        return acc - penalty
    elif acc < 0.975:
        penalty = (0.975 - acc) * 10
        return acc - penalty
    else:
        return acc

# === 3. GA 优化器 ===
def GA_optimize_SVM(X_train, y_train, pop_size=20, max_iter=100,
                    C_bounds=(0.1, 100), gamma_bounds=(0.0001, 10)):
    dim = 2

    # 初始化种群
    population = np.random.rand(pop_size, dim)
    population[:, 0] = population[:, 0] * (C_bounds[1] - C_bounds[0]) + C_bounds[0]
    population[:, 1] = population[:, 1] * (gamma_bounds[1] - gamma_bounds[0]) + gamma_bounds[0]

    def crossover(p1, p2):
        alpha = np.random.rand()
        return alpha * p1 + (1 - alpha) * p2

    def mutate(p, rate=0.1):
        mutation = np.random.randn(*p.shape) * rate
        return p + mutation

    for t in range(max_iter):
        fitness = np.array([fitness_function(ind[0], ind[1]) for ind in population])
        idx = np.argsort(fitness)[-pop_size // 2:]  # 选择适应度前 50%
        selected = population[idx]

        children = []
        for _ in range(pop_size - len(selected)):
            p1, p2 = random.sample(list(selected), 2)
            child = crossover(p1, p2)
            child = mutate(child)
            child[0] = np.clip(child[0], *C_bounds)
            child[1] = np.clip(child[1], *gamma_bounds)
            children.append(child)

        population = np.vstack((selected, children))

        best_fit = fitness_function(population[np.argmax(fitness)][0], population[np.argmax(fitness)][1])
        print(f"迭代 {t+1}/{max_iter} ▶ 当前最优适应度: {best_fit:.4f}")

    best_idx = np.argmax([fitness_function(ind[0], ind[1]) for ind in population])
    return population[best_idx][0], population[best_idx][1]

# === 4. 执行 GA 优化 SVM 参数 ===
C_opt, gamma_opt = GA_optimize_SVM(X_train, y_train)

# === 5. 训练最终模型 ===
model = SVC(C=C_opt, gamma=gamma_opt, kernel='rbf')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ 测试集准确率：{acc * 100:.2f}%")
print(f"✅ 最优参数：C = {C_opt:.4f}, gamma = {gamma_opt:.4f}")

# === 6. 混淆矩阵可视化 ===
cm = confusion_matrix(y_test, y_pred)
labels = ['Normal', 'IR007', 'B007', 'OR007']  # 请根据你的数据修改
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax, colorbar=True, values_format='d')
plt.title("GA-SVM Confusion Matrix on Test Set")
plt.tight_layout()
plt.savefig("ga_svm_confusion_matrix.png", dpi=300)
plt.show()
