import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import random

# === 数据加载（你需要确保路径正确）===
X = np.load('tsne_results/X_tsne_3d.npy')
y = np.load('tsne_results/y_tsne_3d.npy')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42, stratify=y)


def fitness_function(X_train, y_train, C, gamma):
    model = SVC(C=C, gamma=gamma, kernel='rbf')
    scores = cross_val_score(model, X_train, y_train, cv=5)
    penalty_C = 0.0005 * C
    penalty_gamma = 0.01 * gamma  # 你可以根据 gamma 的取值范围调整比例
    return np.mean(scores) - penalty_C - penalty_gamma


# === WOA 优化器 ===
def WOA_optimize_SVM(X_train, y_train, pop_size=20, max_iter=60, C_bounds=(0.1, 100), gamma_bounds=(0.0001, 10)):
    dim = 2
    population = np.random.rand(pop_size, dim)
    population[:, 0] = population[:, 0] * (C_bounds[1] - C_bounds[0]) + C_bounds[0]
    population[:, 1] = population[:, 1] * (gamma_bounds[1] - gamma_bounds[0]) + gamma_bounds[0]

    fitness = np.array([fitness_function(X_train, y_train, ind[0], ind[1]) for ind in population])
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
            population[i, 0] = np.clip(population[i, 0], *C_bounds)
            population[i, 1] = np.clip(population[i, 1], *gamma_bounds)

        fitness = np.array([fitness_function(X_train, y_train, ind[0], ind[1]) for ind in population])
        best_idx = np.argmax(fitness)
        best_position = population[best_idx].copy()
    return best_position[0], best_position[1]

# === PSO 优化器 ===
def PSO_optimize_SVM(X_train, y_train, pop_size=20, max_iter=60, C_bounds=(0.1, 100), gamma_bounds=(0.0001, 10)):
    dim = 2
    w, c1, c2 = 0.7, 1.5, 1.5
    position = np.random.rand(pop_size, dim)
    position[:, 0] = position[:, 0] * (C_bounds[1] - C_bounds[0]) + C_bounds[0]
    position[:, 1] = position[:, 1] * (gamma_bounds[1] - gamma_bounds[0]) + gamma_bounds[0]
    velocity = np.zeros_like(position)

    pbest = position.copy()
    pbest_fitness = np.array([fitness_function(X_train, y_train, ind[0], ind[1]) for ind in position])
    gbest_idx = np.argmax(pbest_fitness)
    gbest = position[gbest_idx].copy()

    for t in range(max_iter):
        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            velocity[i] = w * velocity[i] + c1 * r1 * (pbest[i] - position[i]) + c2 * r2 * (gbest - position[i])
            position[i] += velocity[i]
            position[i, 0] = np.clip(position[i, 0], *C_bounds)
            position[i, 1] = np.clip(position[i, 1], *gamma_bounds)

            fit = fitness_function(X_train, y_train, position[i, 0], position[i, 1])
            if fit > pbest_fitness[i]:
                pbest[i] = position[i].copy()
                pbest_fitness[i] = fit
                if fit > fitness_function(X_train, y_train, gbest[0], gbest[1]):
                    gbest = position[i].copy()
    return gbest[0], gbest[1]

# === GA 优化器 ===
def GA_optimize_SVM(X_train, y_train, pop_size=20, max_iter=60, C_bounds=(0.1, 100), gamma_bounds=(0.0001, 10)):
    def crossover(p1, p2):
        alpha = np.random.rand()
        return alpha * p1 + (1 - alpha) * p2

    def mutate(p, rate=0.1):
        mutation = np.random.randn(*p.shape) * rate
        return p + mutation

    population = np.random.rand(pop_size, 2)
    population[:, 0] = population[:, 0] * (C_bounds[1] - C_bounds[0]) + C_bounds[0]
    population[:, 1] = population[:, 1] * (gamma_bounds[1] - gamma_bounds[0]) + gamma_bounds[0]

    for t in range(max_iter):
        fitness = np.array([fitness_function(X_train, y_train, ind[0], ind[1]) for ind in population])
        idx = np.argsort(fitness)[-pop_size // 2:]
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

    best_idx = np.argmax([fitness_function(X_train, y_train, ind[0], ind[1]) for ind in population])
    return population[best_idx][0], population[best_idx][1]

# === baseline SVM ===
def SVM_baseline(X_train, y_train):
    return 1.0, 0.1

# === 模型训练与评估 ===
def evaluate_and_display(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ {model_name} 准确率: {acc * 100:.2f}%")
    cm = confusion_matrix(y_test, y_pred)
    labels = ['Normal', 'IR007', 'B007', 'OR007']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', ax=ax, colorbar=True, values_format='d')
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png", dpi=300)
    plt.show()

# === 主流程运行 ===
results = {}
for name, optimizer in [
    ("WOA-SVM", WOA_optimize_SVM),
    ("PSO-SVM", PSO_optimize_SVM),
    ("GA-SVM", GA_optimize_SVM),
    ("SVM", SVM_baseline)
]:
    print(f"\n🚀 正在执行: {name}")
    C_opt, gamma_opt = optimizer(X_train, y_train)
    model = SVC(C=C_opt, gamma=gamma_opt, kernel='rbf')
    model.fit(X_train, y_train)
    evaluate_and_display(name, model, X_test, y_test)
    results[name] = (C_opt, gamma_opt)

print("\n✅ 所有模型最优参数:")
for k, v in results.items():
    print(f"{k}: C = {v[0]:.4f}, gamma = {v[1]:.4f}")
