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


def q1(sim_time=1000*ms, dt=0.05*ms, N_pre=1000, N_post=1000,
	   ext_rate_pre=100*Hz, noise_rate_pre=500*Hz, noise_rate_post=500*Hz,
	   fan_in=4, seed=42):
	"""构建 1000 前神经元 + 1000 后神经元网络：
	- 每个后神经元从随机 4 个前神经元接收突触（AMPA, on_pre: s_AMPA_ext_post += 1）
	- 前神经元：100Hz 外部脉冲 + 500Hz 噪声
	- 后神经元：仅 500Hz 噪声 + 来自前神经元的突触输入
	- 输出：后神经元发放图（时间-神经元编号散点），保存到 1_3/q1.png
	"""

	# 随机种子与时间步长
	b2.seed(seed)
	b2.defaultclock.dt = dt

	# 人群：前、后神经元（使用相同方程；后者不会接收外部100Hz脉冲）
	Gpre = b2.NeuronGroup(N_pre, model=exp,
						  threshold='v > v_peak', reset='v = c_reset; u += d_reset',
						  method='euler')
	Gpost = b2.NeuronGroup(N_post, model=exp,
						   threshold='v > v_peak', reset='v = c_reset; u += d_reset',
						   method='euler')
	for G in (Gpre, Gpost):
		G.v = v_r
		G.u = 0*pA
		G.s_AMPA_ext = 0
		G.s_AMPA_noise = 0

	# 外部与噪声输入
	# 前：100Hz 外部脉冲（one-to-one）
	Pext = b2.PoissonGroup(N_pre, rates=ext_rate_pre)
	S_ext = b2.Synapses(Pext, Gpre, on_pre='s_AMPA_ext_post += 1')
	S_ext.connect(j='i')

	# 前：500Hz 噪声（one-to-one）
	Pn_pre = b2.PoissonGroup(N_pre, rates=noise_rate_pre)
	S_noise_pre = b2.Synapses(Pn_pre, Gpre, on_pre='s_AMPA_noise_post += 1')
	S_noise_pre.connect(j='i')

	# 后：500Hz 噪声（one-to-one）
	Pn_post = b2.PoissonGroup(N_post, rates=noise_rate_post)
	S_noise_post = b2.Synapses(Pn_post, Gpost, on_pre='s_AMPA_noise_post += 1')
	S_noise_post.connect(j='i')

	# 前 -> 后 随机固定入度 fan_in（每个后神经元 4 个前驱）
	S_prepost = b2.Synapses(Gpre, Gpost, on_pre='s_AMPA_ext_post += 1')
	# 显式构造连接（确保每个后神经元恰好选到 fan_in 个不同前神经元）
	rng = np.random.default_rng(seed)
	for j in range(N_post):
		idx = rng.choice(N_pre, size=fan_in, replace=False)
		S_prepost.connect(i=idx, j=j)

	# 监视后神经元发放
	Sm_post = b2.SpikeMonitor(Gpost)

	# 运行
	b2.run(sim_time)

	# 绘制发放图（后神经元）
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.scatter(Sm_post.t/ms, Sm_post.i, s=1.5, c='tab:blue', marker='.')
	ax.set_xlabel('时间 (ms)')
	ax.set_ylabel('神经元编号（后）')
	ax.set_xlim(0.0, float(sim_time/ms))
	ax.set_ylim(-0.5, N_post-0.5)
	fig.tight_layout()
	plt.savefig('1_3/q1.png', dpi=200)
	plt.close(fig)
	return Sm_post


if __name__ == '__main__':
	# 中文字体设置（可选）
	plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
	plt.rcParams['axes.unicode_minus'] = False
	q1()
