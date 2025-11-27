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


def q1(sim_time=1000*ms, dt=0.05*ms, ext_rate_pre=100*Hz, noise_rate_pre=500*Hz, noise_rate_post=500*Hz):
	"""两个神经元：
	- 突触前(索引0)：接收 100Hz 外部输入 + 500Hz 噪声
	- 突触后(索引1)：接收来自突触前的脉冲（AMPA）+ 500Hz 噪声
	绘制突触后神经元膜电位 v(t)，保存至 1_2/q1.png
	"""

	b2.defaultclock.dt = dt

	# 两个神经元
	G = b2.NeuronGroup(
		2,
		model=exp,
		threshold='v > v_peak',
		reset='v = c_reset; u += d_reset',
		method='euler',
	)
	G.v = v_r
	G.u = 0*pA
	G.s_AMPA_ext = 0
	G.s_AMPA_noise = 0
	# 已移除单独的突触导通率变量；突触事件直接增加 postsyn 的 s_AMPA_ext

	# 突触前外部输入 100Hz -> 仅作用于索引0
	Pext_pre = b2.PoissonGroup(1, rates=ext_rate_pre)
	S_ext_pre = b2.Synapses(Pext_pre, G, on_pre='s_AMPA_ext_post += 1')
	S_ext_pre.connect(i=0, j=0)

	# 噪声 500Hz -> 前、后分别各自接收
	Pnoise_pre = b2.PoissonGroup(1, rates=noise_rate_pre)
	S_noise_pre = b2.Synapses(Pnoise_pre, G, on_pre='s_AMPA_noise_post += 1')
	S_noise_pre.connect(i=0, j=0)

	Pnoise_post = b2.PoissonGroup(1, rates=noise_rate_post)
	S_noise_post = b2.Synapses(Pnoise_post, G, on_pre='s_AMPA_noise_post += 1')
	S_noise_post.connect(i=0, j=1)

	# 突触前 -> 突触后：复用 AMPA 外部变量，前神经元发放时使后神经元 s_AMPA_ext += 1
	S_pre_post = b2.Synapses(G, G, on_pre='s_AMPA_ext_post += 1')
	S_pre_post.connect(i=0, j=1)

	# 监视突触后神经元膜电位
	M_post = b2.StateMonitor(G, 'v', record=[1])
	Sm_pre = b2.SpikeMonitor(G[:1])  # 可选：记录前神经元发放
	Sm_post = b2.SpikeMonitor(G[1:])  # 可选：记录后神经元发放

	b2.run(sim_time)

	# 绘图：仅绘制突触后 v(t)
	fig, ax = plt.subplots(figsize=(10, 4))
	t = M_post.t / ms
	ax.plot(t, M_post.v[0] / mV, color='steelblue')
	ax.axhline(float(v_peak/mV), color='r', ls='--', lw=1, label='v_peak')
	ax.set_xlim(0.0, float(sim_time/ms))
	ax.set_xlabel('时间 (ms)')
	ax.set_ylabel('突触后膜电位 v (mV)')
	ax.legend(loc='upper right')
	fig.tight_layout()
	plt.savefig('1_2/q1.png', dpi=200)
	plt.close(fig)
	return M_post, Sm_pre, Sm_post


if __name__ == '__main__':
	# 配置中文字体
	plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
	plt.rcParams['axes.unicode_minus'] = False
	q1()