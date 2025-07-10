from sklearn.svm import SVC
from scipy.optimize import minimize
import numpy as np

# 适应度函数：计算当前 SVM 参数的分类准确率
def fitness_function(params, X_train, y_train, X_test, y_test):
    C, gamma = params
    model = SVC(C=C, gamma=gamma, kernel='rbf')
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return -accuracy  # WOA 寻找最小值，所以返回负的准确率

# WOA 参数初始化
def whale_optimization(X_train, y_train, X_test, y_test):
    # 定义 SVM 参数空间：C 和 gamma 的范围
    bounds = [(0.1, 100), (1e-6, 1e-1)]  # C 的范围，gamma 的范围

    # 初始位置：随机生成 C 和 gamma
    initial_params = [np.random.uniform(low=0.1, high=100), np.random.uniform(low=1e-6, high=1e-1)]

    # 使用优化算法求解最优参数
    result = minimize(fitness_function, initial_params, args=(X_train, y_train, X_test, y_test), bounds=bounds, method='L-BFGS-B')

    best_C, best_gamma = result.x
    return best_C, best_gamma



from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 获取鲸鱼优化后的 SVM 参数
best_C, best_gamma = whale_optimization(X_train, y_train, X_test, y_test)

# 使用优化后的参数训练 SVM
svm = SVC(C=best_C, gamma=best_gamma, kernel='rbf')
svm.fit(X_train, y_train)

# 预测并计算准确率
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"优化后的SVM参数: C = {best_C}, gamma = {best_gamma}")
print(f"SVM分类准确率: {accuracy * 100:.2f}%")
