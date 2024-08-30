import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def obj_f(x):
    return 11 * np.sin(x) + 7 * np.cos(5 * x)


# 参数
n = 30
w_max = 0.9  # 惯性
w_min = 0.4
c1 = 2   # 个体
c2 = 2   # 社会
K = 50   # 迭代次数

v_max = 0.6
x_lb = -3
x_ub = 3

# 初始化
x = np.random.uniform(x_lb, x_ub, n)  # 位置
v = np.random.uniform(-v_max, v_max, n)  # 速度
fit = np.array([obj_f(x[i]) for i in range(n)])  # 适应度
p_best = x.copy()  # 个体最优
g_best = x[np.argmax(fit)]  # 全局最优

# 创建图形
fig, ax = plt.subplots()
x_vals = np.linspace(x_lb, x_ub, 400)
y_vals = obj_f(x_vals)
ax.plot(x_vals, y_vals, label='目标函数')
scat = ax.scatter(x, obj_f(x), color='red', label='粒子位置')
ax.legend()

# 更新函数


def update(frame):
    global x, v, p_best, g_best, fit

    for i in range(n):  # n个粒子
        # 更新速度
        w = w_max - (w_max - w_min) * frame / K
        r1, r2 = np.random.rand(2)
        v[i] = w * v[i] + c1 * r1 * (p_best[i] - x[i]) + c2 * r2 * (g_best - x[i])
        # 限速
        v[i] = np.clip(v[i], -v_max, v_max)

        # 更新位置
        x[i] += v[i]
        # 限幅
        x[i] = np.clip(x[i], x_lb, x_ub)

        # 更新适应度
        fit[i] = obj_f(x[i])
        # 更新个体最优
        if fit[i] > obj_f(p_best[i]):
            p_best[i] = x[i]

    # 更新全局最优
    if np.max(fit) > obj_f(g_best):
        g_best = x[np.argmax(fit)]

    # 更新散点图
    scat.set_offsets(np.c_[x, obj_f(x)])
    ax.set_title(f"第{frame + 1}次迭代，当前惯性系数：{w:.4f}，全局最优：{obj_f(g_best):.4f}, x={g_best:.4f}")


# 动画设置
ani = FuncAnimation(fig, update, frames=K, repeat=False)

# 显示动画
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
