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
I_syn = I_AMPA_ext + I_AMPA_noise : amp
I_AMPA_ext = g_AMPA_ext*s_AMPA_ext*(v_E - v) : amp
I_AMPA_noise = g_AMPA_noise*s_AMPA_noise*(v_E - v) : amp
ds_AMPA_ext/dt = -s_AMPA_ext/tau_AMPA : 1
ds_AMPA_noise/dt = -s_AMPA_noise/tau_AMPA : 1
"""

def q2():
	"""1000 个同构神经元 + 500Hz 独立泊松噪音输入，绘制发放图（raster）。

	- 神经元模型与 q1 一致，阈值/重置相同
	- 噪音突触电流与 AMPA 相同公式，电导 g_AMPA_noise = 0.5 nS
	- 每个神经元接收独立 500 Hz 泊松输入（逐个对应 one-to-one）
	- 输出图保存到 1_1/q2.png
	"""

	N = 1000
	sim_time = 1000*ms
	dt = 0.05*ms
	ext_rate = 100*Hz
	noise_rate = 500*Hz

	b2.defaultclock.dt = dt

	G = b2.NeuronGroup(
		N=N,
		model=exp,
		threshold='v > v_peak',
		reset='v = c_reset; u += d_reset',
		method='euler',
	)
	G.v = v_r
	G.u = 0*pA
	G.s_AMPA_ext = 0
	G.s_AMPA_noise = 0

	# 外部 100Hz 泊松输入（逐个对应）
	Pext = b2.PoissonGroup(N, rates=ext_rate)
	S_ext = b2.Synapses(Pext, G, on_pre='s_AMPA_ext_post += 1')
	S_ext.connect(j='i')

	# 噪声 500Hz 泊松输入（逐个对应）
	Pn = b2.PoissonGroup(N, rates=noise_rate)
	S_noise = b2.Synapses(Pn, G, on_pre='s_AMPA_noise_post += 1')
	S_noise.connect(j='i')

	Sm = b2.SpikeMonitor(G)

	b2.run(sim_time)

	# 绘制发放图（raster）
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.scatter(Sm.t/ms, Sm.i, s=1, c='tab:blue', marker='.')
	ax.set_xlabel('时间 (ms)')
	ax.set_ylabel('神经元编号')
	ax.set_xlim(0.0, float(sim_time/ms))
	ax.set_ylim(-0.5, N-0.5)
	fig.tight_layout()
	plt.savefig('1_1/q2.png', dpi=200)
	plt.close(fig)
	return Sm

if __name__ == '__main__':
	# 运行所需配置：100Hz 外部输入，1000ms，总步长 0.05ms
	# 配置中文字体
	plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 黑体、微软雅黑
	plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
	q2()