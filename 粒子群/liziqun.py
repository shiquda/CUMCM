import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 定义目标函数


def obj_fun(x):
    x1, x2 = x
    return x1**2 + x2**2 - x1*x2 - 10*x1 - 4*x2 + 60


# 参数设置
n = 30          # 粒子数量
narvs = 2       # 变量个数
c1 = 2.0        # 个体学习因子
c2 = 2.0        # 社会学习因子
w = 0.4        # 惯性权重
K = 30         # 迭代次数
vmax = [6, 6]   # 最大速度
x_lb = [-15, -15]  # 下界
x_ub = [15, 15]    # 上界

# 初始化粒子的位置和速度
x = np.random.uniform(low=x_lb, high=x_ub, size=(n, narvs))
v = np.random.uniform(low=-np.array(vmax), high=np.array(vmax), size=(n, narvs))

# 计算初始适应度
fit = np.array([obj_fun(x[i, :]) for i in range(n)])
pbest = x.copy()  # 每个粒子的历史最佳位置
gbest = x[np.argmin(fit)]  # 全局最佳位置

# 初始化图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x1 = np.linspace(-15, 15, 100)
x2 = np.linspace(-15, 15, 100)
x1, x2 = np.meshgrid(x1, x2)
y = obj_fun([x1, x2])
ax.plot_surface(x1, x2, y, cmap='viridis', alpha=0.5)
sc = ax.scatter(x[:, 0], x[:, 1], fit, color='r')

# 迭代过程
fitness_best = np.zeros(K)
for d in range(K):
    for i in range(n):
        # 更新速度
        r1, r2 = np.random.rand(2)
        v[i, :] = (w * v[i, :] +
                   c1 * r1 * (pbest[i, :] - x[i, :]) +
                   c2 * r2 * (gbest - x[i, :]))
        # 限制速度
        v[i, :] = np.clip(v[i, :], -np.array(vmax), np.array(vmax))

        # 更新位置
        x[i, :] = x[i, :] + v[i, :]
        # 限制位置
        x[i, :] = np.clip(x[i, :], x_lb, x_ub)

        # 计算适应度
        fit[i] = obj_fun(x[i, :])
        # 更新个体最佳位置
        if fit[i] < obj_fun(pbest[i, :]):
            pbest[i, :] = x[i, :]
        # 更新全局最佳位置
        if fit[i] < obj_fun(gbest):
            gbest = x[i, :]

    fitness_best[d] = obj_fun(gbest)

    # 动态更新散点图
    sc._offsets3d = (x[:, 0], x[:, 1], fit)
    plt.pause(0.1)

plt.figure()
plt.plot(fitness_best)
plt.xlabel('迭代次数')
plt.ylabel('适应度最小值')
plt.show()

print(f"最佳的位置是：{gbest}")
print(f"此时最优值是：{obj_fun(gbest)}")
