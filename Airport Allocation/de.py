import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime, timedelta

# 设置中文字体，避免乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 时间转换
def parse_time(time_str):
    dt = datetime.strptime(time_str, "%m-%d %H:%M")
    base = datetime(1900, 6, 13, 0, 0)
    return (dt - base).total_seconds() / 60

def minutes_to_time(minutes):
    base = datetime(1900, 6, 13, 0, 0)
    dt = base + timedelta(minutes=minutes)
    return dt.strftime("%m-%d %H:%M")

# 计算适应度和详细指标
def compute_fitness_details(X, params):
    gate_indices = np.array([int(round(x)) % params['N'] for x in X])
    z1 = np.sum((params['departure_times'] - params['arrival_times']) * np.array(params['beta'])[gate_indices])
    penalty = 0
    # 时间冲突检查
    stand_map = {}
    for i, j in enumerate(gate_indices):
        stand_map.setdefault(j, []).append(i)
    for lst in stand_map.values():
        conflicts = sum(params['overlap'][a, b] for a in lst for b in lst if a < b)
        penalty += conflicts * 1e6
    # 尺寸违规检查（向量化）
    size_violations = np.sum((params['wingspans'] > np.array(params['wingspan_limits'])[gate_indices]) |
                             (params['lengths'] > np.array(params['length_limits'])[gate_indices]))
    penalty += size_violations * 1e6
    fitness_value = -(z1 - penalty)
    used_near = len(set(gate_indices) & set(params['near_gates']))
    near_util = used_near / params['total_near'] * 100
    return fitness_value, z1, penalty, near_util

def create_fitness(params):
    def fitness(X):
        return compute_fitness_details(X, params)[0]
    return fitness

def create_callback(params, iteration_count, fitness_history, z1_history, penalty_history, near_util_history, z1_stability):
    def callback(Xk, convergence):
        iteration_count[0] += 1
        fitness_value, z1, penalty, near_util = compute_fitness_details(Xk, params)
        fitness_history.append(fitness_value)
        z1_history.append(z1)
        penalty_history.append(penalty)
        near_util_history.append(near_util)
        z1_stability.append(z1)
        print(f"迭代 {iteration_count[0]}: 适应度 = {fitness_value:.2f}, Z1 = {z1:.2f}, "
              f"惩罚值 = {penalty:.2f}, 近机位利用率 = {near_util:.2f}%, 收敛度 = {convergence:.4f}")
        # 收敛条件：惩罚值=0，Z1 连续10次变化小于100分钟
        if len(z1_stability) >= 10 and penalty == 0:
            recent_z1 = z1_stability[-10:]
            if max(recent_z1) - min(recent_z1) < 100:
                print("收敛条件满足：惩罚值=0，Z1 稳定")
                return True
        return False
    return callback

# 主函数
def main():
    iteration_count = [0]  # 使用列表以便在闭包中修改
    fitness_history = []
    z1_history = []
    penalty_history = []
    near_util_history = []
    z1_stability = []

    # 数据准备
    # 机位数据
    try:
        gates_data = pd.read_excel("./停机位.xlsx")
    except FileNotFoundError as e:
        print(f"机位数据文件未找到: {e}")
        return
    gates_data = gates_data.rename(columns={'机位编号': 'gate', '翼展限制': 'wingspan_limit', '身长限制': 'length_limit'})
    # 排除无效近机位（133, 148, 163）
    gates_data = gates_data[~gates_data['gate'].isin([133, 148, 163])].reset_index(drop=True)
    gates = [d for d in gates_data["gate"]]
    beta = [1 if 101 <= d <= 177 else 0 for d in gates_data["gate"]]
    wingspan_limits = [float(d.strip('m')) for d in gates_data["wingspan_limit"]]
    length_limits = [d.strip('m') for d in gates_data["length_limit"]]
    length_limits = [200.0 if d == '/' else float(d) for d in length_limits]
    N = len(gates)  # 223
    print(
        f"机位信息:\n"
        f"{len(gates)}机位编号：{gates}\n"
        f"{len(beta)}远近机位：{beta}\n"
        f"{len(wingspan_limits)}翼展限制：{wingspan_limits}\n"
        f"{len(length_limits)}身长限制：{length_limits}\n"
    )

    # 航班数据
    try:
        flights_data = pd.read_excel("./航班信息.xlsx")
    except FileNotFoundError as e:
        print(f"航班数据文件未找到: {e}")
        return
    flights_data = flights_data.rename(columns={'航班号': 'flight', '计划进港': 'arrival', '计划离港': 'departure', '翼展': 'wingspan', '身长': 'length'})
    M = len(flights_data)
    arrival_times = np.array([parse_time(f) for f in flights_data["arrival"]])
    departure_times = np.array([parse_time(f) for f in flights_data["departure"]])
    wingspans = np.array([f for f in flights_data["wingspan"]])
    lengths = np.array([f for f in flights_data["length"]])
    print(
        f"航班信息:\n"
        f"{len(arrival_times)}进港时间：{arrival_times}\n"
        f"{len(departure_times)}离港时间：{departure_times}\n"
        f"{len(wingspans)}翼展：{wingspans}\n"
        f"{len(lengths)}身长：{lengths}\n"
    )

    # 时间周期和安全间隔
    T_start = min(arrival_times)
    T_end = max(departure_times) + 30
    total_time = T_end - T_start
    delta_T = 30  # 分钟

    # 预计算时间冲突矩阵
    overlap = np.zeros((M, M), dtype=bool)
    for i in range(M):
        for j in range(M):
            if i != j:
                no_overlap = (departure_times[i] + delta_T <= arrival_times[j]) or \
                             (departure_times[j] + delta_T <= arrival_times[i])
                overlap[i, j] = not no_overlap

    # 近机位信息
    near_gates = [g for g, b in zip(gates, beta) if b == 1]
    total_near = len(near_gates)

    # 设置参数
    params = {
        'M': M,
        'N': N,
        'gates': gates,
        'beta': beta,
        'arrival_times': arrival_times,
        'departure_times': departure_times,
        'wingspans': wingspans,
        'lengths': lengths,
        'wingspan_limits': wingspan_limits,
        'length_limits': length_limits,
        'overlap': overlap,
        'near_gates': near_gates,
        'total_near': total_near
    }

    # 调试：验证 params
    print("Params 键：", list(params.keys()))
    if 'N' not in params:
        print("错误：params 中缺少 'N' 键")
        return

    # 创建 fitness 和 callback 函数
    fitness_func = create_fitness(params)
    callback_func = create_callback(params, iteration_count, fitness_history, z1_history, penalty_history, near_util_history, z1_stability)

    # 差分进化优化
    try:
        result = differential_evolution(
            fitness_func,
            bounds=[(0, N) for _ in range(M)],
            strategy='randtobest1exp',
            popsize=30,
            mutation=0.8,
            recombination=0.5,
            maxiter=1000,
            callback=callback_func,
            updating='deferred',
            tol=-1.0,
            polish=True
        )
        print(f"实际迭代代数: {result.nit}")
    except Exception as e:
        print(f"优化过程中发生错误: {e}")
        return

    # 获取最优分配
    best_X = result.x
    gate_assignments = [int(round(x)) % N for x in best_X]
    assignments = [(flights_data.iloc[i]["flight"], gates[gate_assignments[i]],
                    arrival_times[i], departure_times[i]) for i in range(M)]

    # 计算近机位利用率
    utilization = {gate: 0.0 for gate in near_gates}
    for flight, gate, start, end in assignments:
        if gate in near_gates:
            utilization[gate] += (end - start) / total_time
    avg_utilization = sum(utilization.values()) / len(near_gates) if near_gates else 0

    # 生成分配表格
    assignment_data = []
    for flight, gate, start, end in assignments:
        assignment_data.append({
            "Flight Number": flight,
            "Assigned Gate": gate,
            "Arrival Time": minutes_to_time(start),
            "Departure Time": minutes_to_time(end),
            "Near Gate": "Yes" if gate in set(near_gates) else "No"
        })
    df = pd.DataFrame(assignment_data)
    print("\n航班机位预分配表格：")
    print(df.to_string(index=False))
    # 保存带时间戳的CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"DE-gate_assignments_{timestamp}.csv", index=False)

    # 绘制近机位利用率图表（所有近机位）
    plt.figure(figsize=(20, 6))  # 增大宽度以容纳更多编号
    plt.bar(near_gates, [utilization[gate] * 100 for gate in near_gates], color='blue', alpha=0.6)
    plt.axhline(y=avg_utilization * 100, color='red', linestyle='--',
                label=f'平均利用率: {avg_utilization:.2%}')
    plt.xlabel('近机位编号')
    plt.ylabel('利用率 (%)')
    plt.title('DE-近机位利用率 (101-177)')
    plt.xticks(near_gates, rotation=45, ha='right')  # 旋转45度，右对齐
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    # 绘制近机位甘特图
    # 计算 near_gates_indices
    near_gates_indices = [i for i, gate in enumerate(gates) if gate in near_gates]

    # 甘特图绘制
    # 计算 near_gates_indices
    near_gates_indices = [i for i, gate in enumerate(gates) if gate in near_gates]

    # 计算 near_gates_indices
    near_gates_indices = [i for i, gate in enumerate(gates) if gate in near_gates]

    near_assignments = [
        (flight, gate, start, end)
        for flight, gate, start, end in assignments
        if gate in near_gates
    ]

    # 控制整体高度
    fig_height = max(6, len(near_gates) * 0.15)
    plt.figure(figsize=(15, fig_height))

    # bar 高度
    bar_height = 0.2

    for flight, gate, start, end in near_assignments:
        y = near_gates_indices[gates.index(gate)]
        plt.barh(y, end - start, left=start, height=bar_height, color='black', alpha=1)
        plt.barh(y, delta_T, left=end, height=bar_height, color='red', alpha=1)

    # 设置 yticks
    yticks_to_show = list(range(100, 181, 10))
    yticks_labels = []
    yticks_positions = []

    for gate in yticks_to_show:
        if gate in near_gates:
            idx = near_gates_indices[gates.index(gate)]
            yticks_positions.append(idx)
            yticks_labels.append(gate)

    plt.yticks(yticks_positions, yticks_labels, rotation=0)

    plt.xlabel('时间 (分钟，自06-13 00:00)')
    plt.ylabel('近机位编号')
    plt.title('DE-近机位分配甘特图')

    legend_handles = [
        Patch(color='black', label='近机位停放'),
        Patch(color='red', label='安全间隔')
    ]
    plt.legend(handles=legend_handles, loc='upper right')

    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()