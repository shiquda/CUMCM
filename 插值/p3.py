# 三、二维插值的三维展示方法

# -*- coding: utf-8 -*-
"""
演示二维插值。
"""
# -*- coding: utf-8 -*-
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import matplotlib.cm as cm


def func(x, y):
    return (x + y) * np.exp(-5.0 * (x**2 + y**2))


# X-Y轴分为20*20的网格
x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
x, y = np.meshgrid(x, y)  # 20*20的网格数据

fvals = func(x, y)  # 计算每个网格点上的函数值

fig = plt.figure(figsize=(9, 6))

# 绘制第一个子图：原始函数的3D图
ax = plt.subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(x, y, fvals, rstride=2, cstride=2, cmap=cm.coolwarm, linewidth=0.5, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.colorbar(surf, shrink=0.5, aspect=5)

# 二维插值
interp_func = RegularGridInterpolator((x[0, :], y[:, 0]), fvals, method='cubic')

# 计算100*100的网格上的插值
xnew = np.linspace(-1, 1, 100)
ynew = np.linspace(-1, 1, 100)
xnew, ynew = np.meshgrid(xnew, ynew)
fnew = interp_func((xnew, ynew))  # 100*100的插值结果

# 绘制第二个子图：插值后的3D图
ax2 = plt.subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(xnew, ynew, fnew, rstride=2, cstride=2, cmap=cm.coolwarm, linewidth=0.5, antialiased=True)
ax2.set_xlabel('xnew')
ax2.set_ylabel('ynew')
ax2.set_zlabel('fnew(x, y)')
plt.colorbar(surf2, shrink=0.5, aspect=5)

plt.show()
