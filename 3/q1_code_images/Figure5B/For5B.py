import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from brian2 import *
import multiprocessing
import time

# 使用 numpy 作为代码生成目标（避免 Cython 编译）
prefs.codegen.target = "numpy"

def reproduce_figure_3a():
    import example
    print("生成 Figure 3A（零相干度动力学）...")
    
    stimparams = dict(Ton=0.5*second, Toff=1.5*second, mu0=40*Hz, coh=0.0)
    sim_dt = 0.1*ms
    T = 2.0*second
    
    sim = example.Simulation(example.modelparams, stimparams, sim_dt, T)
    sim.run(T, randseed=12345)
    
    spikesE = sim.model.mons['spikesE']
    N_E = example.modelparams['N_E']
    fsel = example.modelparams['fsel']
    N1 = int(fsel * N_E)
    N0 = N_E - 2 * N1
    
    times_A = spikesE.t[np.logical_and(spikesE.i >= N0, spikesE.i < N0 + N1)]
    times_B = spikesE.t[np.logical_and(spikesE.i >= N0 + N1, spikesE.i < N_E)]
    
    def get_rate(spike_times, num_neurons, duration, bin_width=0.05):
        bins = np.arange(0, duration/second + bin_width, bin_width)
        hist, _ = np.histogram(spike_times/second, bins)
        centers = bins[:-1] + bin_width/2.0
        rate = hist / (num_neurons * bin_width)
        return centers, rate

    t_bins, rate_A = get_rate(times_A, N1, T)
    _, rate_B = get_rate(times_B, N1, T)
    
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    
    sample_indices = np.arange(N0, N_E, 5)
    mask = np.isin(spikesE.i, sample_indices)
    axes[0].plot(spikesE.t[mask], spikesE.i[mask], '.', markersize=1, color='k')
    axes[0].set_ylabel('Neuron Index')
    axes[0].set_title('Raster Plot (Trial 1)')
    
    axes[1].plot(t_bins, rate_A, 'r', label='Group A')
    axes[1].plot(t_bins, rate_B, 'b', label='Group B')
    axes[1].set_ylabel('Freq (Hz)')
    axes[1].legend()
    
    t_array = np.arange(0, T/second + 0.05, 0.05)
    input_A = np.zeros_like(t_array)
    input_B = np.zeros_like(t_array)
    mask_stim = (t_array >= 0.5) & (t_array < 1.5)
    input_A[mask_stim] = 40 + np.random.normal(0, 4, np.sum(mask_stim))
    input_B[mask_stim] = 40 + np.random.normal(0, 4, np.sum(mask_stim))
    axes[2].step(t_array, input_A, 'r', alpha=0.5)
    axes[2].step(t_array, input_B, 'b', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('wang2002_fig3a_repro.png')
    print("Saved fig3a.")

def analyze_trial_robust(times_tuple, stim_onset=0.5, threshold=15.0, window_width=0.05, step=0.005, N_subpop=240, trial_id=0):
    times_A, times_B = times_tuple
    duration = 2.0
    t_axis = np.arange(0, duration, step)
    
    # 计算放电率
    def get_rate_trace(spike_times):
        spike_times = np.sort(spike_times)
        t_starts = t_axis - window_width/2
        t_ends = t_axis + window_width/2
        idx_start = np.searchsorted(spike_times, t_starts)
        idx_end = np.searchsorted(spike_times, t_ends)
        return (idx_end - idx_start) / (N_subpop * window_width)

    rA = get_rate_trace(times_A)
    rB = get_rate_trace(times_B)
    
    mask = t_axis >= stim_onset
    t_valid = t_axis[mask]
    rA_valid = rA[mask]
    rB_valid = rB[mask]
    
    cross_A = np.where(rA_valid >= threshold)[0]
    cross_B = np.where(rB_valid >= threshold)[0]
    
    t_A = t_valid[cross_A[0]] if len(cross_A) > 0 else 9999
    t_B = t_valid[cross_B[0]] if len(cross_B) > 0 else 9999
    
    # 处理未决策或平局的情况
    if t_A == 9999 and t_B == 9999:
        return None, None # Undecided
        
    # 如果A和B同时达到阈值
    if t_A == t_B:
        # 比较瞬时放电率决定胜负
        rate_at_crossing_A = rA_valid[cross_A[0]]
        rate_at_crossing_B = rB_valid[cross_B[0]]
        
        if rate_at_crossing_A > rate_at_crossing_B:
            return 'A', (t_A - stim_onset) * 1000
        elif rate_at_crossing_B > rate_at_crossing_A:
            return 'B', (t_B - stim_onset) * 1000
        else:
            # 极罕见的完全平局，随机选择
            return ('A' if np.random.rand() > 0.5 else 'B'), (t_A - stim_onset) * 1000

    # 正常判定
    if t_A < t_B:
        return 'A', (t_A - stim_onset) * 1000
    else:
        return 'B', (t_B - stim_onset) * 1000

def run_single_trial_task(args):
    coh, seed, stimparams, modelparams, sim_dt, max_duration, trial_idx = args
    import brian2
    brian2.device.reinit()
    brian2.start_scope()
    brian2.prefs.codegen.target = "numpy"

    try:
        import example
        sim = example.Simulation(modelparams, stimparams, sim_dt, max_duration)
        sim.run(max_duration, randseed=seed)
        spikesE = sim.model.mons['spikesE']

        N_E = modelparams['N_E']
        fsel = modelparams['fsel']
        N_subpop = int(N_E * fsel)
        N_nonselect = N_E - 2 * N_subpop

        idx_start_A = N_nonselect
        idx_end_A = N_nonselect + N_subpop
        idx_start_B = idx_end_A
        idx_end_B = idx_end_A + N_subpop

        times_A = np.array(spikesE.t[np.logical_and(spikesE.i >= idx_start_A, spikesE.i < idx_end_A)] / second)
        times_B = np.array(spikesE.t[np.logical_and(spikesE.i >= idx_start_B, spikesE.i < idx_end_B)] / second)
        
        return (times_A, times_B, N_subpop, trial_idx)

    except Exception as e:
        print(f"Error: {e}")
        return None

def reproduce_figure_5b_final():
    print("Starting Figure 5B Reproduction (Calibrated Version)...")
    import example
    
    coherences = [3.2, 6.4, 12.8, 25.6, 51.2] 
    
    # 增加试验次数以平滑曲线
    trials_per_coh = 100  # 样本量设为100以保证统计稳定
    
    final_acc = []
    final_rt = []
    final_rt_err = []
    tasks = []
    base_seed = 12345
    
    # 调整输入强度以匹配论文中的反应时间范围
    # 使用40Hz作为基准输入
    adjusted_mu0 = 40 * Hz 

    print(f"Config: mu0={adjusted_mu0}, Trials/coh={trials_per_coh}")

    for coh in coherences:
        stimparams = dict(Ton=0.5*second, Toff=2.0*second, mu0=adjusted_mu0, coh=coh)
        for i in range(trials_per_coh):
            seed = base_seed + int(coh*10000) + i
            # 减小时间步长以提高精度
            # 0.05ms精度更高
            tasks.append((coh, seed, stimparams, example.modelparams, 0.05*ms, 2.0*second, i))
            
    # 并行运行试验
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    results_raw = pool.map(run_single_trial_task, tasks)
    pool.close()
    pool.join()
    
    print("模拟完成，开始分析数据...")
    
    data_by_coh = {c: {'correct': 0, 'total': 0, 'rts': []} for c in coherences}
    
    for i, res in enumerate(results_raw):
        if res is None: continue
        task_coh = tasks[i][0]
        times_A, times_B, N_sub, t_id = res
        
        # 阈值设为 15Hz
        choice, rt = analyze_trial_robust((times_A, times_B), 
                                          stim_onset=0.5, 
                                          threshold=15.0, 
                                          N_subpop=N_sub)
        
        if choice is not None:
            data_by_coh[task_coh]['total'] += 1
            data_by_coh[task_coh]['rts'].append(rt)
            if choice == 'A':
                data_by_coh[task_coh]['correct'] += 1
                
    # 输出统计结果
    print("-" * 40)
    for coh in coherences:
        d = data_by_coh[coh]
        n = d['total']
        acc = (d['correct'] / n * 100) if n > 0 else 0
        mean_rt = np.mean(d['rts']) if n > 0 else 0
        std_rt = np.std(d['rts']) if n > 0 else 0
        
        final_acc.append(acc)
        final_rt.append(mean_rt)
        final_rt_err.append(std_rt)
        print(f"Coh {coh:4.1f}%: Acc={acc:5.1f}%, RT={mean_rt:5.1f} ms (n={n})")
    print("-" * 40)

    # 绘制图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 神经测量函数（准确率）
    ax1.plot(coherences, final_acc, 'o-', color='k', markerfacecolor='k')
    ax1.set_xscale('log')
    ax1.set_xticks([3.2, 6.4, 12.8, 25.6, 51.2])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_ylim(40, 105)
    ax1.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Coherence (%)')
    ax1.set_ylabel('% Correct')
    ax1.set_title('Neurometric Function')
    ax1.grid(True, which='both', linestyle='--', alpha=0.3)

    # 2. 计时函数（反应时）
    ax2.errorbar(coherences, final_rt, yerr=final_rt_err, fmt='o-', color='k', capsize=5)
    ax2.set_xscale('log')
    ax2.set_xticks([3.2, 6.4, 12.8, 25.6, 51.2])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # 设定 Y 轴范围
    ax2.set_ylim(0, 1000) 
    ax2.set_xlabel('Coherence (%)')
    ax2.set_ylabel('Reaction Time (ms)')
    ax2.set_title('Chronometric Function')
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure5b_re.png')
    print("Saved figure5b_re.png")

if __name__ == '__main__':
    reproduce_figure_5b_final()