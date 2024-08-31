"""
演示二维插值。
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pylab as pl
import matplotlib as mpl


def func(x, y):
    return (x + y) * np.exp(-5.0 * (x**2 + y**2))


# X-Y轴分为15*15的网格
y = np.linspace(-1, 1, 15)
x = np.linspace(-1, 1, 15)
x_grid, y_grid = np.meshgrid(x, y)

fvals = func(x_grid, y_grid)  # 计算每个网格点上的函数值

# 使用 RegularGridInterpolator 进行插值
interp_func = RegularGridInterpolator((x, y), fvals, method='cubic')

# 计算100*100的网格上的插值
xnew = np.linspace(-1, 1, 100)
ynew = np.linspace(-1, 1, 100)
xnew_grid, ynew_grid = np.meshgrid(xnew, ynew)
fnew = interp_func((xnew_grid, ynew_grid))

# 绘图
pl.subplot(121)
im1 = pl.imshow(fvals, extent=[-1, 1, -1, 1], cmap=mpl.cm.hot, interpolation='nearest', origin="lower")
pl.colorbar(im1)

pl.subplot(122)
im2 = pl.imshow(fnew, extent=[-1, 1, -1, 1], cmap=mpl.cm.hot, interpolation='nearest', origin="lower")
pl.colorbar(im2)
pl.show()
