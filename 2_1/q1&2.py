import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.lines import Line2D
from pathlib import Path

# ----------------------
# 参数与分段函数 g(h)
# ----------------------
alpha = 1.0
wEE = 1.5
h1_ext = 0.8
h2_ext = 0.8

def g(h):
    """分段函数 g(h)"""
    if h < -0.2:
        return 0.0
    elif h < 0.2:
        return 0.1 + 0.5*h
    elif h < 0.8:
        return h
    elif h < 1.2:
        return 0.4 + 0.5*h
    else:
        return 1.0

# 向量化版本（真正基于数组运算）
def g_arr(h):
    h = np.asarray(h)
    return np.where(
        h < -0.2, 0.0,
        np.where(
            h < 0.2, 0.1 + 0.5*h,
            np.where(
                h < 0.8, h,
                np.where(h < 1.2, 0.4 + 0.5*h, 1.0)
            )
        )
    )

def rhs(h_vec):
    """系统右端：dh/dt = f(h)"""
    h1, h2 = h_vec
    dh1 = -h1 + (wEE - alpha)*g(h1) - alpha*g(h2) + h1_ext
    dh2 = -h2 + (wEE - alpha)*g(h2) - alpha*g(h1) + h2_ext
    return np.array([dh1, dh2])

# ----------------------
# 1. 用 fsolve 寻找平衡点
# ----------------------
# 给几个不同的初值，避免只找到一个根
guesses = [
    (0.5, 0.5),    # 对称态
    (1.2, -0.2),   # 1 高 2 低
    (-0.2, 1.2),   # 1 低 2 高
    (0.0, 0.0),
]

sols = []
for guess in guesses:
    sol, info, ier, msg = fsolve(lambda x: rhs(x), guess, full_output=True)
    if ier == 1:   # 收敛
        # 检查是否与已有解“不同”（按距离去重）
        if not any(np.linalg.norm(sol - s) < 1e-3 for s in sols):
            sols.append(sol)

print("找到的平衡点：")
for s in sols:
    print(s)

# ----------------------
# 2. 画零斜线与相图
# ----------------------
# 网格范围与分辨率（可调大以获得更平滑的等高线）
N = 500
h1_vals = np.linspace(-0.5, 1.5, N)
h2_vals = np.linspace(-0.5, 1.5, N)
H1, H2 = np.meshgrid(h1_vals, h2_vals)

# 计算网格上的 dh1/dt, dh2/dt（数组运算更快更平滑）
DH1 = -H1 + (wEE - alpha)*g_arr(H1) - alpha*g_arr(H2) + h1_ext
DH2 = -H2 + (wEE - alpha)*g_arr(H2) - alpha*g_arr(H1) + h2_ext

plt.figure(figsize=(7, 6))

# dh1/dt = 0 与 dh2/dt = 0 的零斜线
cs1 = plt.contour(H1, H2, DH1, levels=[0.0], colors='C0', linewidths=1.5)
cs2 = plt.contour(H1, H2, DH2, levels=[0.0], colors='C1', linestyles='--', linewidths=1.5)

# 向量场箭头（略微降低密度：调大抽稀步长）
step = max(20, N // 20)
plt.quiver(H1[::step, ::step], H2[::step, ::step],
           DH1[::step, ::step], DH2[::step, ::step],
           angles='xy')

# 平衡点用点标出来
for s in sols:
    plt.plot(s[0], s[1], 'o')
    plt.text(s[0] + 0.03, s[1] + 0.03,
             f'({s[0]:.2f}, {s[1]:.2f})')

plt.xlabel(r'$h_1$')
plt.ylabel(r'$h_2$')
plt.title('Nullclines and phase portrait')
# 通过代理对象创建图例，避免直接访问 QuadContourSet.collections
legend_handles = [
    Line2D([0], [0], color='C0', lw=1.5, label=r'$dh_1/dt=0$'),
    Line2D([0], [0], color='C1', lw=1.5, linestyle='--', label=r'$dh_2/dt=0$'),
]
plt.legend(handles=legend_handles)
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
# 将图片保存到脚本所在目录
out_path = Path(__file__).parent / 'phase_portrait.png'
plt.savefig(out_path.as_posix(), dpi=300, bbox_inches='tight')
plt.close()
