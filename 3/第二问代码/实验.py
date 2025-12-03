import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 模型参数定义 [cite: 843]
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
    'mu0': 35.0       # Hz (老师建议值)
}

# 模拟设置
dt = 0.0005        # 0.5 ms 时间步长
sim_duration = 3.0 # 模拟总时长 (秒)
time_steps = int(sim_duration / dt)
t_axis = np.linspace(0, sim_duration, time_steps)

# ==========================================
# 2. 核心数学函数实现 [cite: 838-842]
# ==========================================
def transfer_func(I, p):
    """
    非线性传递函数 F(I) [cite: 838]
    公式: r = (aI - b) / (1 - exp(-d*(aI - b)))
    """
    val = p['a'] * I - p['b']
    arg = -p['d'] * val
    
    # 数值稳定性处理：防止 exp 溢出或除以 0
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        r = val / (1 - np.exp(arg))
    
    # 修正 NaN/Inf 或负值
    if np.isscalar(r):
        if np.isnan(r) or np.isinf(r) or r < 0: r = 0.0
    else:
        r[np.isnan(r) | np.isinf(r) | (r < 0)] = 0.0
        
    return r

def run_trial(coherence, p, seed=None):
    """
    运行单次试验
    """
    if seed is not None:
        np.random.seed(seed)
        
    # 初始化状态变量 (s: 门控变量, Ib: 噪声电流)
    s1, s2 = 0.001, 0.001
    Ib1, Ib2 = p['I0'], p['I0']
    
    # 记录轨迹
    r1_trace = np.zeros(time_steps)
    r2_trace = np.zeros(time_steps)
    
    # 设置输入信号强度
    # 假设 c' = coherence/100
    u1 = p['mu0'] * (1 + coherence / 100.0)
    u2 = p['mu0'] * (1 - coherence / 100.0)
    
    # 刺激时间窗口 (0.5s ~ 1.5s)
    idx_on = int(0.5 / dt)
    idx_off = int(1.5 / dt)
    
    # 预计算噪声参数 (Euler-Maruyama 方法)
    # 噪声项 variance = 1/dt [老师建议]
    # dIb = ... + sqrt(sigma^2 / tau_0) * dW
    # dW ~ N(0, dt) = sqrt(dt) * N(0, 1)
    noise_scale = np.sqrt(p['sigma']**2 / p['tau_0']) * np.sqrt(dt)
    decay_factor = dt / p['tau_0']
    
    # 决策阈值 (Hz)
    threshold = 20.0 
    decision_choice = None
    decision_time = None
    
    for i in range(time_steps):
        t = i * dt
        
        # 1. 施加外部刺激
        curr_u1 = u1 if (i >= idx_on and i < idx_off) else 0.0
        curr_u2 = u2 if (i >= idx_on and i < idx_off) else 0.0
        
        # 2. 计算突触总电流 [cite: 839]
        # I = gE*s_self - gI*s_opp + Ib + gext*u
        I1 = p['gE'] * s1 - p['gI'] * s2 + Ib1 + p['gext'] * curr_u1
        I2 = p['gE'] * s2 - p['gI'] * s1 + Ib2 + p['gext'] * curr_u2
        
        # 3. 计算放电率 r [cite: 838]
        r1 = transfer_func(I1, p)
        r2 = transfer_func(I2, p)
        
        r1_trace[i] = r1
        r2_trace[i] = r2
        
        # 4. 更新门控变量 s [cite: 841]
        s1 += dt * (p['gamma'] * r1 * (1 - s1) - s1 / p['tau_s'])
        s2 += dt * (p['gamma'] * r2 * (1 - s2) - s2 / p['tau_s'])
        
        # 5. 更新噪声电流 Ib [cite: 842]
        # 注意：这里严格遵循离散化白噪声的处理
        Ib1 += - (Ib1 - p['I0']) * decay_factor + noise_scale * np.random.normal()
        Ib2 += - (Ib2 - p['I0']) * decay_factor + noise_scale * np.random.normal()
        
        # 6. 判定决策 (仅在刺激开始后)
        if decision_choice is None and i >= idx_on:
            if r1 >= threshold:
                decision_choice = 'A'
                decision_time = (t - 0.5) * 1000 # ms
            elif r2 >= threshold:
                decision_choice = 'B'
                decision_time = (t - 0.5) * 1000 # ms
                
    return t_axis, r1_trace, r2_trace, decision_choice, decision_time

# ==========================================
# 3. 主程序：绘图与保存
# ==========================================
def main():
    print("开始运行简化放电率模型模拟...")
    
    # ------------------------------------------------
    # 任务 1: 生成并保存放电率动态图 (Figure Dynamics)
    # ------------------------------------------------
    coh_example = 6.4
    print(f"正在生成动态图 (Coherence={coh_example}%)...")
    t, r1, r2, choice, rt = run_trial(coh_example, params, seed=42)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, r1, 'r-', linewidth=2, label='Population 1 (Preferred)')
    plt.plot(t, r2, 'b-', linewidth=2, label='Population 2 (Null)')
    # 绘制刺激区间
    plt.axvspan(0.5, 1.5, color='gray', alpha=0.1, label='Stimulus Period')
    plt.axhline(20.0, color='k', linestyle='--', label='Decision Threshold (20Hz)')
    
    plt.title(f'Simplified Rate Model Dynamics (Coherence = {coh_example}%)')
    plt.xlabel('Time (s)')
    plt.ylabel('Firing Rate (Hz)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 保存图片
    filename_dyn = 'rate_model_dynamics.png'
    plt.savefig(filename_dyn, dpi=300)
    print(f"-> 已保存: {filename_dyn}")
    plt.close() # 关闭图形释放内存

    # ------------------------------------------------
    # 任务 2: 统计准确率与反应时间 (Batch Simulation)
    # ------------------------------------------------
    coherences = [3.2, 6.4, 12.8, 25.6, 51.2]
    n_trials = 100 # 每个相干度模拟次数
    
    results_acc = []
    results_rt = []
    results_rt_err = []
    
    print(f"\n开始批量模拟 (每组 {n_trials} 次)...")
    
    for coh in coherences:
        correct_count = 0
        rts = []
        
        for n in range(n_trials):
            # 运行模拟 (不固定随机种子)
            _, _, _, choice, dt_val = run_trial(coh, params)
            
            if choice is not None:
                rts.append(dt_val)
                # 假设 Pop 1 (A) 是正确选项
                if choice == 'A':
                    correct_count += 1
        
        # 计算统计量
        acc = correct_count / n_trials * 100
        mean_rt = np.mean(rts) if len(rts) > 0 else 0
        std_rt = np.std(rts) if len(rts) > 0 else 0
        
        results_acc.append(acc)
        results_rt.append(mean_rt)
        results_rt_err.append(std_rt)
        
        print(f"Coh {coh}%: Acc={acc:.1f}%, Mean RT={mean_rt:.1f}ms")

    # ------------------------------------------------
    # 任务 3: 生成并保存准确率图 (Accuracy)
    # ------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.plot(coherences, results_acc, 'o-', color='black', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xticks(coherences, [str(c) for c in coherences]) # 自定义刻度标签
    plt.ylim(40, 105)
    plt.axhline(50, color='gray', linestyle='--', linewidth=1)
    
    plt.xlabel('Coherence (%)')
    plt.ylabel('% Correct')
    plt.title('Neurometric Function (Accuracy)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    filename_acc = 'rate_model_accuracy.png'
    plt.savefig(filename_acc, dpi=300)
    print(f"-> 已保存: {filename_acc}")
    plt.close()

    # ------------------------------------------------
    # 任务 4: 生成并保存反应时间图 (Reaction Time)
    # ------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.errorbar(coherences, results_rt, yerr=results_rt_err, fmt='o-', color='black', 
                 linewidth=2, capsize=5, markersize=8)
    plt.xscale('log')
    plt.xticks(coherences, [str(c) for c in coherences])
    
    # 根据结果自动调整范围，或者参考论文 0-1000ms
    plt.ylim(0, max(results_rt) * 1.5) 
    
    plt.xlabel('Coherence (%)')
    plt.ylabel('Reaction Time (ms)')
    plt.title('Chronometric Function (Reaction Time)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    filename_rt = 'rate_model_rt.png'
    plt.savefig(filename_rt, dpi=300)
    print(f"-> 已保存: {filename_rt}")
    plt.close()
    
    print("\n所有图像生成完毕。")

if __name__ == "__main__":
    main()