import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 固定随机种子，保证可复现
np.random.seed(42)

# 适应度函数（用训练集拆验证集的验证准确率）
def fitness_function(X_train, y_train, C, gamma):
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


# PSO优化SVM
def PSO_optimize_SVM(X_train, y_train, pop_size=20, max_iter=100, C_bounds=(0.1, 100), gamma_bounds=(0.0001, 10)):
    dim = 2  # C和gamma两个维度
    w = 0.7  # 惯性权重
    c1 = 1.5  # 个体学习因子
    c2 = 1.5  # 社会学习因子

    # 初始化粒子位置和速度
    population = np.random.rand(pop_size, dim)
    population[:, 0] = population[:, 0] * (C_bounds[1] - C_bounds[0]) + C_bounds[0]
    population[:, 1] = population[:, 1] * (gamma_bounds[1] - gamma_bounds[0]) + gamma_bounds[0]
    velocity = np.zeros((pop_size, dim))

    # 个体历史最好位置和适应度
    pbest_position = population.copy()
    pbest_fitness = np.array([fitness_function(X_train, y_train, C=ind[0], gamma=ind[1]) for ind in population])

    # 群体最好位置和适应度
    gbest_idx = np.argmax(pbest_fitness)
    gbest_position = pbest_position[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    for t in range(max_iter):
        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            velocity[i] = (w * velocity[i]
                           + c1 * r1 * (pbest_position[i] - population[i])
                           + c2 * r2 * (gbest_position - population[i]))

            population[i] += velocity[i]

            # 边界约束
            population[i, 0] = np.clip(population[i, 0], C_bounds[0], C_bounds[1])
            population[i, 1] = np.clip(population[i, 1], gamma_bounds[0], gamma_bounds[1])

            # 计算适应度
            fitness = fitness_function(X_train, y_train, C=population[i, 0], gamma=population[i, 1])

            # 更新个体历史最优
            if fitness > pbest_fitness[i]:
                pbest_position[i] = population[i].copy()
                pbest_fitness[i] = fitness

                # 更新群体最优
                if fitness > gbest_fitness:
                    gbest_position = population[i].copy()
                    gbest_fitness = fitness

        print(f"迭代 {t+1}/{max_iter} ▶ 最优验证准确率 = {gbest_fitness:.4f} | C = {gbest_position[0]:.4f}, γ = {gbest_position[1]:.4f}")

    return gbest_position[0], gbest_position[1]

# 主流程
if __name__ == "__main__":
    X = np.load('tsne_results/X_tsne_3d.npy')
    y = np.load('tsne_results/y_tsne_3d.npy')

    # 固定训练测试划分随机种子
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42, stratify=y)

    # 执行PSO优化
    C_opt, gamma_opt = PSO_optimize_SVM(X_train, y_train, pop_size=20, max_iter=100)

    # 用训练集全部数据训练最终模型
    final_model = SVC(C=C_opt, gamma=gamma_opt, kernel='rbf')
    final_model.fit(X_train, y_train)

    # 测试集评估
    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n✅ 测试集准确率：", round(acc * 100, 2), "%")
    print("✅ 最优参数：C =", round(C_opt, 4), ", gamma =", round(gamma_opt, 4))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    labels = ['Normal', 'IR007', 'B007', 'OR007']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', ax=ax, colorbar=True, values_format='d')
    plt.title("PSO-SVM Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.savefig("pso_svm_confusion_matrix.png", dpi=300)
    plt.show()
