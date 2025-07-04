import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime, timedelta

# 设置中文字体，避免乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


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
    stand_map = {}
    for i, j in enumerate(gate_indices):
        stand_map.setdefault(j, []).append(i)
    for lst in stand_map.values():
        conflicts = sum(params['overlap'][a, b] for a in lst for b in lst if a < b)
        penalty += conflicts * 1e6
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


# 粒子群优化算法
def particle_swarm_optimization(fitness_func, M, N, params, pop_size=100, max_iter=100, w=0.7, c1=1.5, c2=1.5):
    np.random.seed(42)
    X = np.random.uniform(0, N, size=(pop_size, M))
    V = np.zeros_like(X)
    pbest = X.copy()
    pbest_values = np.array([fitness_func(x) for x in X])
    gbest_idx = np.argmin(pbest_values)
    gbest = pbest[gbest_idx].copy()
    gbest_value = pbest_values[gbest_idx]
    iteration_count = 0

    fitness_history = []
    z1_stability = []

    while iteration_count < max_iter:
        for i in range(pop_size):
            r1 = np.random.uniform(0, 1, M)
            r2 = np.random.uniform(0, 1, M)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
            X[i] += V[i]
            X[i] = np.clip(X[i], 0, N)

            fitness_value = fitness_func(X[i])
            if fitness_value < pbest_values[i]:
                pbest[i] = X[i].copy()
                pbest_values[i] = fitness_value
                if fitness_value < gbest_value:
                    gbest = X[i].copy()
                    gbest_value = fitness_value

        iteration_count += 1
        fval, z1, penalty, near_util = compute_fitness_details(gbest, params)
        fitness_history.append(fval)
        z1_stability.append(z1)
        print(
            f"迭代 {iteration_count}: 适应度 = {fval:.2f}, Z1 = {z1:.2f}, 惩罚 = {penalty:.2f}, 近机位利用率 = {near_util:.2f}%")

        if penalty == 0 and len(z1_stability) >= 10:
            recent_z1 = z1_stability[-10:]
            if max(recent_z1) - min(recent_z1) < 100:
                print("收敛条件满足：惩罚值=0，Z1 稳定")
                break
    return gbest, gbest_value


# 主函数
def main():
    try:
        gates_data = pd.read_excel("./停机位.xlsx")
    except FileNotFoundError as e:
        print(f"机位数据文件未找到: {e}")
        return
    gates_data = gates_data.rename(
        columns={'机位编号': 'gate', '翼展限制': 'wingspan_limit', '身长限制': 'length_limit'})
    gates_data = gates_data[~gates_data['gate'].isin([133, 148, 163])].reset_index(drop=True)
    gates = [d for d in gates_data["gate"]]
    beta = [1 if 101 <= d <= 177 else 0 for d in gates_data["gate"]]
    wingspan_limits = [float(d.strip('m')) for d in gates_data["wingspan_limit"]]
    length_limits = [d.strip('m') for d in gates_data["length_limit"]]
    length_limits = [200.0 if d == '/' else float(d) for d in length_limits]
    N = len(gates)

    try:
        flights_data = pd.read_excel("./航班信息.xlsx")
    except FileNotFoundError as e:
        print(f"航班数据文件未找到: {e}")
        return
    flights_data = flights_data.rename(
        columns={'航班号': 'flight', '计划进港': 'arrival', '计划离港': 'departure', '翼展': 'wingspan',
                 '身长': 'length'})
    M = len(flights_data)
    arrival_times = np.array([parse_time(f) for f in flights_data["arrival"]])
    departure_times = np.array([parse_time(f) for f in flights_data["departure"]])
    wingspans = np.array([f for f in flights_data["wingspan"]])
    lengths = np.array([f for f in flights_data["length"]])

    T_start = min(arrival_times)
    T_end = max(departure_times) + 30
    total_time = T_end - T_start
    delta_T = 30

    overlap = np.zeros((M, M), dtype=bool)
    for i in range(M):
        for j in range(M):
            if i != j:
                no_overlap = (departure_times[i] + delta_T <= arrival_times[j]) or \
                             (departure_times[j] + delta_T <= arrival_times[i])
                overlap[i, j] = not no_overlap

    near_gates = [g for g, b in zip(gates, beta) if b == 1]
    total_near = len(near_gates)

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

    fitness_func = create_fitness(params)
    gbest, gbest_value = particle_swarm_optimization(fitness_func, M, N, params, pop_size=30, max_iter=1000)

    gate_assignments = [int(round(x)) % N for x in gbest]
    assignments = [(flights_data.iloc[i]["flight"], gates[gate_assignments[i]],
                    arrival_times[i], departure_times[i]) for i in range(M)]

    utilization = {gate: 0.0 for gate in near_gates}
    for flight, gate, start, end in assignments:
        if gate in near_gates:
            utilization[gate] += (end - start) / total_time
    avg_utilization = sum(utilization.values()) / len(near_gates) if near_gates else 0

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"POS-gate_assignments_PSO_{timestamp}.csv", index=False)

    plt.figure(figsize=(20, 6))
    plt.bar(near_gates, [utilization[gate] * 100 for gate in near_gates], color='blue', alpha=0.6)
    plt.axhline(y=avg_utilization * 100, color='red', linestyle='--',
                label=f'平均利用率: {avg_utilization:.2%}')
    plt.xlabel('近机位编号')
    plt.ylabel('利用率 (%)')
    plt.title('POS-近机位利用率 (101-177)')
    plt.xticks(near_gates, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    near_gates_indices = [i for i, gate in enumerate(gates) if gate in near_gates]

    near_assignments = [
        (flight, gate, start, end)
        for flight, gate, start, end in assignments
        if gate in near_gates
    ]

    fig_height = max(6, len(near_gates) * 0.15)
    plt.figure(figsize=(15, fig_height))
    bar_height = 0.2

    for flight, gate, start, end in near_assignments:
        y = near_gates_indices[gates.index(gate)]
        plt.barh(y, end - start, left=start, height=bar_height, color='black', alpha=1)
        plt.barh(y, delta_T, left=end, height=bar_height, color='red', alpha=1)

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
    plt.title('POS-近机位分配甘特图')

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
