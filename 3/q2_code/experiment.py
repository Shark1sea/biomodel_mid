import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 模型参数 [cite: 843]
# ==========================================
params = {
    'a': 270.0,       # Hz/nA
    'b': 108.0,       # Hz
    'd': 0.154,       # sec
    'gE': 0.2609,     # nA (self-excitation)
    'gI': 0.0497,     # nA (cross-inhibition)
    'gext': 0.00052,  # nA
    'gamma': 0.641,   # unitless
    'tau_s': 0.1,     # sec (100ms)
    'tau_0': 0.002,   # sec (2ms)
    'I0': 0.325,      # nA
    'sigma': 0.02,    # nA
    'mu0': 35.0       # Hz
}

# 模拟设置
dt = 0.0005        # 0.5 ms 时间步长
sim_duration = 3.0 # 模拟总时长（秒）
time_steps = int(sim_duration / dt)
T_AXIS = np.linspace(0, sim_duration, time_steps)

# ==========================================
# 2. 核心函数 [cite: 838-842]
# ==========================================

def transfer_func(I, p):
    """非线性传递函数 F(I) [cite: 838]
    r = (aI - b) / (1 - exp(-d*(aI - b)))
    """
    val = p['a'] * I - p['b']
    arg = -p['d'] * val
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        r = val / (1 - np.exp(arg))
    if np.isscalar(r):
        if np.isnan(r) or np.isinf(r) or r < 0: r = 0.0
    else:
        r[np.isnan(r) | np.isinf(r) | (r < 0)] = 0.0
    return r


def run_trial(coherence, p, seed=None):
    if seed is not None:
        np.random.seed(seed)
    s1, s2 = 0.001, 0.001
    Ib1, Ib2 = p['I0'], p['I0']
    r1_trace = np.zeros(time_steps)
    r2_trace = np.zeros(time_steps)
    u1 = p['mu0'] * (1 + coherence / 100.0)
    u2 = p['mu0'] * (1 - coherence / 100.0)
    idx_on = int(0.5 / dt)
    idx_off = int(1.5 / dt)
    noise_scale = np.sqrt(p['sigma']**2 / p['tau_0']) * np.sqrt(dt)
    decay_factor = dt / p['tau_0']
    threshold = 20.0  # 决策阈值（Hz）
    decision_choice = None
    decision_time = None
    for i in range(time_steps):
        t = i * dt
        curr_u1 = u1 if (i >= idx_on and i < idx_off) else 0.0
        curr_u2 = u2 if (i >= idx_on and i < idx_off) else 0.0
        I1 = p['gE'] * s1 - p['gI'] * s2 + Ib1 + p['gext'] * curr_u1
        I2 = p['gE'] * s2 - p['gI'] * s1 + Ib2 + p['gext'] * curr_u2
        r1 = transfer_func(I1, p)
        r2 = transfer_func(I2, p)
        r1_trace[i] = r1
        r2_trace[i] = r2
        s1 += dt * (p['gamma'] * r1 * (1 - s1) - s1 / p['tau_s'])
        s2 += dt * (p['gamma'] * r2 * (1 - s2) - s2 / p['tau_s'])
        Ib1 += - (Ib1 - p['I0']) * decay_factor + noise_scale * np.random.normal()
        Ib2 += - (Ib2 - p['I0']) * decay_factor + noise_scale * np.random.normal()
        if decision_choice is None and i >= idx_on:
            if r1 >= threshold:
                decision_choice = 'A'
                decision_time = (t - 0.5) * 1000  # 相对刺激起始的毫秒
            elif r2 >= threshold:
                decision_choice = 'B'
                decision_time = (t - 0.5) * 1000  # 相对刺激起始的毫秒
    return T_AXIS, r1_trace, r2_trace, decision_choice, decision_time


def main():
    print("开始运行简化速率模型...")
    coh_example = 6.4
    print(f"生成动力学图（相干度={coh_example}%）...")
    t, r1, r2, choice, rt = run_trial(coh_example, params, seed=42)
    plt.figure(figsize=(10, 6))
    plt.plot(t, r1, 'r-', linewidth=2, label='Population 1 (Preferred)')
    plt.plot(t, r2, 'b-', linewidth=2, label='Population 2 (Null)')
    plt.axvspan(0.5, 1.5, color='gray', alpha=0.1, label='刺激区间')
    plt.axhline(20.0, color='k', linestyle='--', label='决策阈值 (20Hz)')
    plt.title(f'简化速率模型动力学（相干度 = {coh_example}%）')
    plt.xlabel('时间 (s)')
    plt.ylabel('发放率 (Hz)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    filename_dyn = 'rate_model_dynamics.png'
    plt.savefig(filename_dyn, dpi=300)
    print(f"-> 已保存: {filename_dyn}")
    plt.close()

    coherences = [3.2, 6.4, 12.8, 25.6, 51.2]
    n_trials = 100
    results_acc = []
    results_rt = []
    results_rt_err = []
    print(f"\n开始批量模拟（每个相干度 {n_trials} 次）...")
    for coh in coherences:
        correct_count = 0
        rts = []
        for n in range(n_trials):
            _, _, _, choice, dt_val = run_trial(coh, params)
            if choice is not None:
                rts.append(dt_val)
                if choice == 'A':
                    correct_count += 1
        acc = correct_count / n_trials * 100
        mean_rt = np.mean(rts) if len(rts) > 0 else 0
        std_rt = np.std(rts) if len(rts) > 0 else 0
        results_acc.append(acc)
        results_rt.append(mean_rt)
        results_rt_err.append(std_rt)
        print(f"Coh {coh}%: Acc={acc:.1f}%, Mean RT={mean_rt:.1f}ms")

    plt.figure(figsize=(6, 5))
    plt.plot(coherences, results_acc, 'o-', color='black', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xticks(coherences, [str(c) for c in coherences])
    plt.ylim(40, 105)
    plt.axhline(50, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('相干度 (%)')
    plt.ylabel('正确率 (%)')
    plt.title('神经测量函数（准确率）')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    filename_acc = 'rate_model_accuracy.png'
    plt.savefig(filename_acc, dpi=300)
    print(f"-> 已保存: {filename_acc}")
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.errorbar(coherences, results_rt, yerr=results_rt_err, fmt='o-', color='black', 
                 linewidth=2, capsize=5, markersize=8)
    plt.xscale('log')
    plt.xticks(coherences, [str(c) for c in coherences])
    plt.ylim(0, max(results_rt) * 1.5)
    plt.xlabel('相干度 (%)')
    plt.ylabel('反应时间 (ms)')
    plt.title('计时函数（反应时间）')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    filename_rt = 'rate_model_rt.png'
    plt.savefig(filename_rt, dpi=300)
    print(f"-> 已保存: {filename_rt}")
    plt.close()
    print("\n所有图像已生成。")


if __name__ == "__main__":
    main()
