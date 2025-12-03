# 2-1

## 建模对象

兴奋性神经元群动力学：这是一个二维耦合的互相抑制/自抑制系统，形式与 Wilson–Cowan 模型相近。g(h) 是分段饱和的激活函数，代表群体发放率的非线性响应。

## 求解方式

### 公式

$$
\begin{align}
\frac{dh_{E,1}}{dt} = -h_{E,1} + (w_{EE} - \alpha)g_E(h_{E,1}) - \alpha g_E(h_{E_2}) + h_1^{ext} \\
\frac{dh_{E,2}}{dt} = -h_{E,2} + (w_{EE} - \alpha)g_E(h_{E,2}) - \alpha g_E(h_{E_1}) + h_2^{ext} \\

\end{align}
$$

### 固定点求解

令
$$
\begin{align}
\frac{dh_1}{dt} = 0 \\
\frac{dh_2}{dt} = 0 \\
\end{align}
$$

使用数值根求解（fsolve）配合多初值去重，得到可能的多个平衡点

### 零斜线与相图

零斜线交点即为固定点，叠加矢量场以判断流向和吸引子结构

### 稳定性判断

雅可比矩阵
$J=\begin{pmatrix}
-1 + (w_{EE}-\alpha)\,g'(h_1) & -\alpha\,g'(h_2) \\
-\alpha\,g'(h_1) & -1 + (w_{EE}-\alpha)\,g'(h_2)
\end{pmatrix}$

$g'(h)$取对应分段的斜率，通过特征值符合判断稳定性（负实部为稳定）

**相图中用颜色区分稳定性：**

- 绿色圆点：稳定（吸引子）
- 灰色圆点：鞍点
- 红色圆点：不稳定
