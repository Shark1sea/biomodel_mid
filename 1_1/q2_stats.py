import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
from brian2.units import amp, volt, farad, mA, uA, nA, pA, mV, uF, nF, pF, second, ms, us, siemens, mS, uS, nS, Hz

# 参数（与 q2 保持一致）
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
    """计算群体与个体统计指标。
    返回：dict 包含 rate_per_neuron, mean_rate, std_rate, isi_cv_per_neuron, fano_factor_per_neuron 等。
    """
    # 每个神经元的发放次数
    counts = np.bincount(spike_indices, minlength=N)
    duration_s = sim_time_ms / 1000.0
    rate_per_neuron = counts / duration_s  # Hz
    mean_rate = float(np.mean(rate_per_neuron))
    std_rate = float(np.std(rate_per_neuron))

    # ISI 与 CV（对每个神经元单独计算）
    isi_cv_per_neuron = np.full(N, np.nan)
    fano_factor_per_neuron = np.full(N, np.nan)

    # 为了近似 Fano，按窗口统计：将时间分成 10 个等宽窗口
    n_windows = 10
    edges = np.linspace(0.0, sim_time_ms, n_windows + 1)

    for i in range(N):
        ti = spike_times_ms[spike_indices == i]
        if ti.size >= 2:
            isi = np.diff(ti)
            mu = np.mean(isi)
            sigma = np.std(isi)
            if mu > 0:
                isi_cv_per_neuron[i] = sigma / mu
        # Fano 因子：窗口内 spike 计数的方差/均值
        win_counts, _ = np.histogram(ti, bins=edges)
        m = np.mean(win_counts)
        v = np.var(win_counts)
        if m > 0:
            fano_factor_per_neuron[i] = v / m

    return {
        'rate_per_neuron': rate_per_neuron,
        'mean_rate': mean_rate,
        'std_rate': std_rate,
        'isi_cv_per_neuron': isi_cv_per_neuron,
        'fano_factor_per_neuron': fano_factor_per_neuron,
    }


def q2_stats():
    """1000 个同构神经元 + 500Hz 独立泊松噪音输入，计算群体统计并绘图保存。

    生成：
    - 栅格图：`1_1/q2_stats_raster.png`
    - 发放率直方图：`1_1/q2_stats_rate_hist.png`
    - ISI CV 分布：`1_1/q2_stats_isi_cv_hist.png`
    - Fano 因子分布：`1_1/q2_stats_fano_hist.png`
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

    Pext = b2.PoissonGroup(N, rates=ext_rate)
    S_ext = b2.Synapses(Pext, G, on_pre='s_AMPA_ext_post += 1')
    S_ext.connect(j='i')

    Pn = b2.PoissonGroup(N, rates=noise_rate)
    S_noise = b2.Synapses(Pn, G, on_pre='s_AMPA_noise_post += 1')
    S_noise.connect(j='i')

    Sm = b2.SpikeMonitor(G)

    b2.run(sim_time)

    # 收集发放数据
    spike_times_ms = Sm.t / ms
    spike_indices = np.array(Sm.i, dtype=int)

    stats = compute_stats(spike_times_ms, spike_indices, N, float(sim_time/ms))

    # 栅格图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(spike_times_ms, spike_indices, s=1, c='tab:blue', marker='.')
    ax.set_xlabel('时间 (ms)')
    ax.set_ylabel('神经元编号')
    ax.set_xlim(0.0, float(sim_time/ms))
    ax.set_ylim(-0.5, N-0.5)
    fig.tight_layout()
    plt.savefig('1_1/q2_stats_raster.png', dpi=200)
    plt.close(fig)

    # 发放率直方图
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(stats['rate_per_neuron'], bins=40, color='steelblue', alpha=0.85)
    ax.set_xlabel('发放率 (Hz)')
    ax.set_ylabel('神经元数')
    ax.set_title(f"均值={stats['mean_rate']:.2f} Hz,  标准差={stats['std_rate']:.2f} Hz")
    fig.tight_layout()
    plt.savefig('1_1/q2_stats_rate_hist.png', dpi=200)
    plt.close(fig)

    # ISI CV 分布
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = stats['isi_cv_per_neuron'][~np.isnan(stats['isi_cv_per_neuron'])]
    ax.hist(vals, bins=40, color='darkorange', alpha=0.85)
    ax.set_xlabel('ISI CV')
    ax.set_ylabel('神经元数')
    ax.set_title(f"有效样本={vals.size}")
    fig.tight_layout()
    plt.savefig('1_1/q2_stats_isi_cv_hist.png', dpi=200)
    plt.close(fig)

    # Fano 因子分布
    fig, ax = plt.subplots(figsize=(8, 5))
    fano_vals = stats['fano_factor_per_neuron'][~np.isnan(stats['fano_factor_per_neuron'])]
    ax.hist(fano_vals, bins=40, color='seagreen', alpha=0.85)
    ax.set_xlabel('Fano 因子')
    ax.set_ylabel('神经元数')
    ax.set_title(f"有效样本={fano_vals.size}")
    fig.tight_layout()
    plt.savefig('1_1/q2_stats_fano_hist.png', dpi=200)
    plt.close(fig)

    # 控制台摘要
    print(f"群体平均发放率: {stats['mean_rate']:.3f} Hz  (std={stats['std_rate']:.3f})")
    if vals.size > 0:
        print(f"ISI CV 平均: {np.mean(vals):.3f}  中位数: {np.median(vals):.3f}")
    if fano_vals.size > 0:
        print(f"Fano 因子平均: {np.mean(fano_vals):.3f}  中位数: {np.median(fano_vals):.3f}")

    return stats


if __name__ == '__main__':
    # 配置中文字体（可选）
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    q2_stats()
