# -*- coding: utf-8 -*-
"""
学生代码模板：计算伽马函数 Gamma(a)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import factorial, sqrt, pi, exp, log

# --- Task 1: 绘制被积函数 ---

def integrand_gamma(x, a):
    """
    计算伽马函数的原始被积函数: f(x, a) = x^(a-1) * exp(-x)
    """
    # 数组输入
    if isinstance(x, np.ndarray):
        res = np.exp((a-1)*np.log(x) - x)
        res = np.where(x<0, 0.0, res)
        # x==0 特殊
        if a == 1:
            res = np.where(x==0, 1.0, res)
        elif a > 1:
            res = np.where(x==0, 0.0, res)
        else:
            res = np.where(x==0, np.inf, res)
        return res

    # 标量输入
    if x < 0:
        return 0.0
    if x == 0:
        if a == 1:
            return 1.0
        elif a > 1:
            return 0.0
        else:
            return np.inf
    try:
        return exp((a-1)*log(x) - x)
    except ValueError:
        return np.nan

def plot_integrands():
    x_vals = np.linspace(0.01, 10, 400)
    plt.figure(figsize=(10, 6))
    for a_val in [2, 3, 4]:
        y_vals = np.array([integrand_gamma(x, a_val) for x in x_vals])
        plt.plot(x_vals, y_vals, label=f'$a={a_val}$')
        peak_x = a_val - 1
        if peak_x > 0:
            plt.plot(peak_x, integrand_gamma(peak_x, a_val), 'o')
    plt.xlabel("$x$")
    plt.ylabel("$x^{a-1}e^{-x}$")
    plt.title("Integrand of Gamma")
    plt.legend()
    plt.grid(True)
    plt.ylim(0)
    plt.xlim(0)
    plt.show()

# --- Task 4: 变换与积分 ---

def transformed_integrand_gamma(z, a):
    # a<=1 时不使用此变换
    if a <= 1.0:
        return 0.0
    c = a - 1.0
    if z <= 0 or z >= 1:
        return 0.0
    x = c * z / (1 - z)
    dxdz = c / (1 - z)**2
    val = integrand_gamma(x, a) * dxdz
    return val if np.isfinite(val) else 0.0

def gamma_function(a):
    if a <= 0:
        return np.nan
    if a > 1.0:
        val, _ = quad(transformed_integrand_gamma, 0, 1, args=(a,))
    else:
        val, _ = quad(integrand_gamma, 0, np.inf, args=(a,))
    return val

# --- 测试并输出 ---

if __name__ == "__main__":
    # 测试示例
    print("Gamma(1.5) =", gamma_function(1.5), "| exact =", 0.5*sqrt(pi))
    for n in [3,6,10]:
        print(f"Gamma({n}) =", gamma_function(n), "| (n-1)! =", factorial(n-1))

