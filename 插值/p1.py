# 一、一维插值

# -*-coding:utf-8 -*-
import numpy as np
from scipy import interpolate
import pylab as pl

import scienceplots

pl.style.use(['science', 'grid', 'cjk-sc-font', 'no-latex'])

x = np.linspace(0, 10, 11)
# x=[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]
y = np.sin(x)
xnew = np.linspace(0, 10, 101)
pl.plot(x, y, "ro")

for kind in ["nearest", "zero", "slinear", "quadratic", "cubic"]:  # 插值方式
    # "nearest","zero"为阶梯插值
    # slinear 线性插值
    # "quadratic","cubic" 为2阶、3阶B样条曲线插值
    f = interpolate.interp1d(x, y, kind=kind)
    # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
    ynew = f(xnew)
    pl.plot(xnew, ynew, label=str(kind))
pl.legend(loc="lower right")
pl.show()
