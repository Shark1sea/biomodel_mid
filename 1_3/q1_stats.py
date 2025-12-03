import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
from brian2.units import ms, Hz, pA, mV, nS, pF

# 复用 1_3/q1.py 的参数与方程（为独立脚本，这里拷贝必要部分）
# 使用带单位的常量，避免单位检查报错
C = 100*pF
v_r = -60*mV
v_t = -40*mV
k = 0.7*pA/(mV**2)
a = 0.03/ms
b = -2*nS
v_peak = 0*mV
c_reset = -50*mV
d_reset = 100*pA
v_E = 0*mV
g_AMPA_ext = 10*nS
g_AMPA_noise = 0.5*nS
tau_AMPA = 2*ms

exp = """
dv/dt = (k*(v - v_r)*(v - v_t) - u + I_syn)/C : volt
du/dt = a*(b*(v - v_r) - u) : amp
I_syn = I_AMPA_ext + I_AMPA_noise : amp
I_AMPA_ext = g_AMPA_ext*s_AMPA_ext*(v_E - v) : amp
I_AMPA_noise = g_AMPA_noise*s_AMPA_noise*(v_E - v) : amp
ds_AMPA_ext/dt = -s_AMPA_ext/tau_AMPA : 1
ds_AMPA_noise/dt = -s_AMPA_noise/tau_AMPA : 1
"""


def compute_stats(spike_times_ms, spike_indices, N, sim_time_ms):
    counts = np.bincount(spike_indices, minlength=N)
    duration_s = sim_time_ms / 1000.0
    rate_per_neuron = counts / duration_s  # Hz
    mean_rate = float(np.mean(rate_per_neuron))
    std_rate = float(np.std(rate_per_neuron))

    isi_cv = np.full(N, np.nan)
    fano = np.full(N, np.nan)

    n_windows = 10
    edges = np.linspace(0.0, sim_time_ms, n_windows + 1)

    for i in range(N):
        ti = spike_times_ms[spike_indices == i]
        if ti.size >= 2:
            isi = np.diff(ti)
            mu = np.mean(isi)
            sig = np.std(isi)
            if mu > 0:
                isi_cv[i] = sig / mu
        win_counts, _ = np.histogram(ti, bins=edges)
        m = np.mean(win_counts)
        v = np.var(win_counts)
        if m > 0:
            fano[i] = v / m

    return {
        'rate_per_neuron': rate_per_neuron,
        'mean_rate': mean_rate,
        'std_rate': std_rate,
        'isi_cv': isi_cv,
        'fano': fano,
    }


def q1_stats(sim_time=1000*ms, dt=0.05*ms, N_pre=1000, N_post=1000,
             ext_rate_pre=100*Hz, noise_rate_pre=500*Hz, noise_rate_post=500*Hz,
             fan_in=4, seed=42):
    """在 1_3/q1 的网络设置基础上，对后神经元群体统计并绘图保存。
    输出图片：
    - 1_3/q1_stats_raster.png（后神经元光栅图）
    - 1_3/q1_stats_rate_hist.png（后神经元发放率直方图）
    - 1_3/q1_stats_isi_cv_hist.png（ISI CV 分布）
    - 1_3/q1_stats_fano_hist.png（Fano 因子分布）
    """
    # 使用 numpy 代码生成以避免 Cython 编译提示
    b2.prefs.codegen.target = "numpy"
    b2.seed(seed)
    b2.defaultclock.dt = dt

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

    # 前：外部与噪声（one-to-one）
    Pext = b2.PoissonGroup(N_pre, rates=ext_rate_pre)
    S_ext = b2.Synapses(Pext, Gpre, on_pre='s_AMPA_ext_post += 1')
    S_ext.connect(j='i')

    Pn_pre = b2.PoissonGroup(N_pre, rates=noise_rate_pre)
    S_noise_pre = b2.Synapses(Pn_pre, Gpre, on_pre='s_AMPA_noise_post += 1')
    S_noise_pre.connect(j='i')

    # 后：噪声（one-to-one）
    Pn_post = b2.PoissonGroup(N_post, rates=noise_rate_post)
    S_noise_post = b2.Synapses(Pn_post, Gpost, on_pre='s_AMPA_noise_post += 1')
    S_noise_post.connect(j='i')

    # 前 -> 后：每个后神经元固定入度 fan_in
    S_prepost = b2.Synapses(Gpre, Gpost, on_pre='s_AMPA_ext_post += 1')
    rng = np.random.default_rng(seed)
    for j in range(N_post):
        idx = rng.choice(N_pre, size=fan_in, replace=False)
        S_prepost.connect(i=idx, j=j)

    # 监视后神经元发放
    Sm_post = b2.SpikeMonitor(Gpost)

    b2.run(sim_time)

    # 收集发放数据
    spike_times_ms = Sm_post.t / ms
    spike_indices = np.array(Sm_post.i, dtype=int)

    stats = compute_stats(spike_times_ms, spike_indices, N_post, float(sim_time/ms))

    # 栅格图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(spike_times_ms, spike_indices, s=1.5, c='tab:blue', marker='.')
    ax.set_xlabel('时间 (ms)')
    ax.set_ylabel('神经元编号（后）')
    ax.set_xlim(0.0, float(sim_time/ms))
    ax.set_ylim(-0.5, N_post-0.5)
    fig.tight_layout()
    plt.savefig('1_3/q1_stats_raster.png', dpi=200)
    plt.close(fig)

    # 发放率直方图
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(stats['rate_per_neuron'], bins=40, color='steelblue', alpha=0.85)
    ax.set_xlabel('发放率 (Hz)')
    ax.set_ylabel('神经元数')
    ax.set_title(f"均值={stats['mean_rate']:.2f} Hz, 标准差={stats['std_rate']:.2f} Hz")
    fig.tight_layout()
    plt.savefig('1_3/q1_stats_rate_hist.png', dpi=200)
    plt.close(fig)

    # ISI CV 分布
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = stats['isi_cv'][~np.isnan(stats['isi_cv'])]
    ax.hist(vals, bins=40, color='darkorange', alpha=0.85)
    ax.set_xlabel('ISI CV')
    ax.set_ylabel('神经元数')
    ax.set_title(f"有效样本={vals.size}")
    fig.tight_layout()
    plt.savefig('1_3/q1_stats_isi_cv_hist.png', dpi=200)
    plt.close(fig)

    # Fano 因子分布
    fig, ax = plt.subplots(figsize=(8, 5))
    fano_vals = stats['fano'][~np.isnan(stats['fano'])]
    ax.hist(fano_vals, bins=40, color='seagreen', alpha=0.85)
    ax.set_xlabel('Fano 因子')
    ax.set_ylabel('神经元数')
    ax.set_title(f"有效样本={fano_vals.size}")
    fig.tight_layout()
    plt.savefig('1_3/q1_stats_fano_hist.png', dpi=200)
    plt.close(fig)

    # 控制台摘要
    print(f"后神经元群体平均发放率: {stats['mean_rate']:.3f} Hz  (std={stats['std_rate']:.3f})")
    if vals.size > 0:
        print(f"ISI CV 平均: {np.mean(vals):.3f}  中位数: {np.median(vals):.3f}")
    if fano_vals.size > 0:
        print(f"Fano 因子平均: {np.mean(fano_vals):.3f}  中位数: {np.median(fano_vals):.3f}")

    return stats


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    q1_stats()
