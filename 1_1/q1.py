import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
from brian2.units import amp, volt, farad, mA, uA, nA, pA, mV, uF, nF, pF, second, ms, us, siemens, mS, uS, nS, Hz

# 参数
C = 100*pF # 膜电容
v_r = -60*mV # 静息电位
v_t = -40*mV # 阈值电位
k = 0.7*pA/(mV**2)
a = 0.03/ms
b = -2*nS
v_peak = 0*mV  # 峰值电位
c_reset = -50*mV
d_reset = 100*pA
v_E = 0*mV
g_AMPA_ext = 10*nS
g_AMPA_noise = 0.5*nS
tau_AMPA = 2*ms

# 模型方程
exp = """
dv/dt = (k*(v - v_r)*(v - v_t) - u + I_syn)/C : volt
du/dt = a*(b*(v - v_r) - u) : amp
I_syn = I_AMPA_ext : amp
I_AMPA_ext = g_AMPA_ext*s_AMPA_ext*(v_E - v) : amp
ds_AMPA_ext/dt = -s_AMPA_ext/tau_AMPA : 1
"""


def q1():
	"""模拟外部泊松脉冲输入为 100Hz 时单神经元的动态，输出三分图。

	三个子图：
	1. 膜电压 v 随时间 (黄色虚线为脉冲发放时刻)
	2. 变量 u 随时间
	3. 有效外部 AMPA 导通率 g_AMPA_ext * s_AMPA_ext 随时间

	配置：
	- 时间步长 dt = 0.05 ms
	- 总模拟时长 sim_time = 1000 ms
	- 外部输入：PoissonGroup，发放率 ext_rate = 100 Hz
	- 发放与重置：v > v_peak 触发；v = c_reset；u += d_reset
	图片保存路径：1_1/q1.png
	"""
	sim_time = 1000*ms
	ext_rate = 100*Hz
	dt = 0.05*ms

	# 设置全局时间步长
	b2.defaultclock.dt = dt

	# 单个神经元，使用上述动力学方程
	G = b2.NeuronGroup(
		N=1,
		model=exp,
		threshold='v > v_peak',
		reset='v = c_reset; u += d_reset',
		method='euler',
	)
	G.v = v_r
	G.u = 0*pA
	G.s_AMPA_ext = 0

	# 外部输入：泊松脉冲（固定 100Hz）
	P = b2.PoissonGroup(1, rates=ext_rate)

	# 每个外部脉冲使后突触神经元的 s_AMPA_ext 加 1
	S_ext = b2.Synapses(P, G, on_pre='s_AMPA_ext_post += 1')
	S_ext.connect()

	# 监视器（记录 v, u, s_AMPA_ext）
	M = b2.StateMonitor(G, variables=['v', 'u', 's_AMPA_ext'], record=True)
	SmG = b2.SpikeMonitor(G)

	# 运行
	b2.run(sim_time)

	# 绘图：仅 v(t)
	fig, ax = plt.subplots(figsize=(10, 4))
	t = M.t / ms
	ax.plot(t, M.v[0] / mV, color='steelblue')
	ax.axhline(float(v_peak/mV), color='r', ls='--', lw=1, label='v_peak')
	for st in SmG.t/ms:
		ax.axvline(st, color='tab:orange', ls='--', lw=1.5, alpha=0.95, zorder=10)
	ax.axvline(-1, color='tab:orange', ls='--', lw=1.5, alpha=0.95, label='脉冲', zorder=10)
	ax.set_xlim(0.0, float(sim_time/ms))
	ax.set_ylabel('膜电压 v (mV)')
	ax.set_xlabel('时间 (ms)')
	ax.legend(loc='upper right')
	fig.tight_layout()
	plt.savefig('1_1/q1.png', dpi=200)
	plt.close(fig)
	return M, SmG

if __name__ == '__main__':
	# 运行所需配置：100Hz 外部输入，1000ms，总步长 0.05ms
	# 配置中文字体
	plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 黑体、微软雅黑
	plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
	q1()