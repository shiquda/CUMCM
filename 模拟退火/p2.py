# 旅行商问题

import numpy as np
import random
import math
from matplotlib import pyplot as plt

# 示例城市坐标 (10 个城市)
cities = [
    (0, 0), (1, 5), (2, 3), (5, 2), (6, 6),
    (8, 3), (7, 9), (3, 8), (4, 4), (9, 5)
]


# 计算距离

def cal_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def total_distance(route):
    res = 0
    for i in range(len(route)):
        c1 = cities[route[i]]
        c2 = cities[route[(i + 1) % len(route)]]
        res += cal_distance(c1, c2)

    return res


def generate_init_sol(n):
    return list(np.random.permutation(n))


def generate_neighbor(route):
    n = len(route)
    i = random.randint(0, n - 1)
    j = random.randint(0, n - 1)
    while i == j:
        j = random.randint(0, n - 1)

    new_route = route.copy()
    new_route[i], new_route[j] = new_route[j], new_route[i]

    return new_route


def acceptance_probability(old_dis, new_dis, t):
    if new_dis < old_dis:
        return 1
    else:
        return math.exp((old_dis - new_dis) / t)


def SA(init_temperature, cooling_rate, max_iterations):
    n = len(cities)

    current_sol = generate_init_sol(n)
    current_dis = total_distance(current_sol)
    best_sol = current_sol
    best_dis = current_dis
    t = init_temperature

    for i in range(max_iterations):
        # 生成邻域解
        neighbor = generate_neighbor(current_sol)
        neighbor_dis = total_distance(neighbor)
        p = acceptance_probability(current_dis, neighbor_dis, t)
        if random.random() < p:
            current_sol = neighbor
            current_dis = neighbor_dis
            if neighbor_dis < best_dis:  # 如果找到更优解
                best_sol = neighbor
                best_dis = neighbor_dis
        t *= cooling_rate
        print(f'第 {i} 次迭代，温度 {t:.4f}，当前最优解 {best_dis:.4f}')
    return best_sol, best_dis


best_sol, best_dis = SA(init_temperature=100, cooling_rate=0.9999, max_iterations=50000)

# 绘图

plt.figure()
plt.scatter(*zip(*cities), c='r', marker='o')
for i, city in enumerate(cities):
    plt.text(city[0], city[1], str(i))

for i in range(len(best_sol)):
    c1 = cities[best_sol[i]]
    c2 = cities[best_sol[(i + 1) % len(best_sol)]]
    plt.plot([c1[0], c2[0]], [c1[1], c2[1]], 'b')

plt.show()
print(f'最优解 {[sol.item() for sol in best_sol]}，最优距离 {best_dis:.4f}')
