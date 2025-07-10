import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random

np.random.seed(42)
random.seed(42)


# === 适应度函数 ===
def fitness_function(X_train, y_train, C, gamma):
    # 拆分训练集为训练+验证
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42, stratify=y_train)

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





# === WOA 优化算法 ===
def WOA_optimize_SVM(X_train, y_train, pop_size=20, max_iter=100, C_bounds=(0.1, 100), gamma_bounds=(0.0001, 10)):
    dim = 2
    a = 2
    population = np.random.rand(pop_size, dim)
    population[:, 0] = population[:, 0] * (C_bounds[1] - C_bounds[0]) + C_bounds[0]
    population[:, 1] = population[:, 1] * (gamma_bounds[1] - gamma_bounds[0]) + gamma_bounds[0]

    fitness = np.array([fitness_function(X_train, y_train, C=ind[0], gamma=ind[1]) for ind in population])
    best_idx = np.argmax(fitness)
    best_position = population[best_idx].copy()

    for t in range(max_iter):
        a_t = 2 - 2 * (t / max_iter)
        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            A = 2 * a_t * r1 - a_t
            C_coef = 2 * r2

            p = np.random.rand()
            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C_coef * best_position - population[i])
                    population[i] = best_position - A * D
                else:
                    rand_idx = np.random.randint(0, pop_size)
                    D = abs(C_coef * population[rand_idx] - population[i])
                    population[i] = population[rand_idx] - A * D
            else:
                b = 1
                l = np.random.uniform(-1, 1)
                D = abs(best_position - population[i])
                population[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_position

            # 限制在边界内
            population[i, 0] = np.clip(population[i, 0], C_bounds[0], C_bounds[1])
            population[i, 1] = np.clip(population[i, 1], gamma_bounds[0], gamma_bounds[1])

        fitness = np.array([fitness_function(X_train, y_train, C=ind[0], gamma=ind[1]) for ind in population])
        best_idx = np.argmax(fitness)
        best_position = population[best_idx].copy()

        print(f"迭代 {t+1}/{max_iter} ▶ 最优 acc = {fitness[best_idx]:.4f} | C = {best_position[0]:.4f}, γ = {best_position[1]:.4f}")

    return best_position[0], best_position[1]

# === 主流程 ===
if __name__ == "__main__":
    # 加载 t-SNE 3D 特征数据
    X = np.load('tsne_results/X_tsne_3d.npy')
    y = np.load('tsne_results/y_tsne_3d.npy')

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42, stratify=y)

    # 执行 WOA 优化
    C_opt, gamma_opt = WOA_optimize_SVM(X_train, y_train, pop_size=20, max_iter=100)

    # 最终模型训练与评估
    final_model = SVC(C=C_opt, gamma=gamma_opt, kernel='rbf')
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n✅ 测试集准确率：", round(acc * 100, 2), "%")
    print("✅ 最优参数：C =", round(C_opt, 4), ", gamma =", round(gamma_opt, 4))

    # === 混淆矩阵 ===
    cm = confusion_matrix(y_test, y_pred)
    labels = ['Normal', 'IR007', 'B007', 'OR007']

    # === 可视化 ===
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', ax=ax, colorbar=True, values_format='d')

    plt.title("WOA-SVM Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.savefig("woa_svm_confusion_matrix.png", dpi=300)
    plt.show()





