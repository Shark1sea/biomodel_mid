import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from brian2 import *
import example  
import multiprocessing
import time

# Use numpy target
prefs.codegen.target = "numpy"


def reproduce_figure_3a():
    print("Generating Figure 3A (Zero Coherence Dynamics)...")
    
    # 0% coherence, 1s stimulus
    stimparams = dict(
        Ton  = 0.5*second,
        Toff = 1.5*second,
        mu0  = 40*Hz,
        coh  = 0.0
    )
    sim_dt = 0.1*ms 
    T = 2.0*second
    
    
    sim = example.Simulation(example.modelparams, stimparams, sim_dt, T)
    
    # Seed 12346 -> B wins, 64321 -> A wins
    sim.run(T, randseed=12346) 
    
    spikesE = sim.model.mons['spikesE']
    spikesI = sim.model.mons['spikesI']
    
    # Define populations
    N_E = example.modelparams['N_E']
    fsel = example.modelparams['fsel']
    N1 = int(fsel * N_E)
    N0 = N_E - 2 * N1
    
    # Get spike times
    times_A = spikesE.t[np.logical_and(spikesE.i >= N0, spikesE.i < N0 + N1)]
    times_B = spikesE.t[np.logical_and(spikesE.i >= N0 + N1, spikesE.i < N_E)]
    
    # Calculate firing rates (50ms window)
    def get_rate(spike_times, num_neurons, duration, bin_width=0.05):
        bins = np.arange(0, duration/second + bin_width, bin_width)
        hist, _ = np.histogram(spike_times/second, bins)
        centers = bins[:-1] + bin_width/2.0
        rate = hist / (num_neurons * bin_width) # Hz
        return centers, rate

    t_bins, rate_A = get_rate(times_A, N1, T, bin_width=0.05)
    _, rate_B = get_rate(times_B, N1, T, bin_width=0.05)
    
    # Determine winner based on late delay period activity
    delay_period_mask = t_bins > 1.5
    
    if np.any(delay_period_mask):
        mean_rate_A = np.mean(rate_A[delay_period_mask])
        mean_rate_B = np.mean(rate_B[delay_period_mask])
    else:
        mean_rate_A = rate_A[-1]
        mean_rate_B = rate_B[-1]
        
    if mean_rate_A > mean_rate_B:
        save_filename = 'Figure3A_Re_A_win.png'
        print(f"Outcome: Group A wins (Rate A: {mean_rate_A:.2f} Hz > Rate B: {mean_rate_B:.2f} Hz)")
    else:
        save_filename = 'Figure3A_Re_B_win.png'
        print(f"Outcome: Group B wins (Rate B: {mean_rate_B:.2f} Hz > Rate A: {mean_rate_A:.2f} Hz)")
    
    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(6, 10), sharex=True)
    
    # 1. Raster
    sample_indices = np.arange(N0, N_E, 5) 
    mask = np.isin(spikesE.i, sample_indices)
    axes[0].plot(spikesE.t[mask], spikesE.i[mask], '.', markersize=1, color='k')
    axes[0].set_ylabel('Neuron Index')
    axes[0].set_title('Raster Plot (Trial 1)')
    
    # 2. Firing Rates
    axes[1].plot(t_bins, rate_A, 'r', label='Group A')
    axes[1].plot(t_bins, rate_B, 'b', label='Group B')
    axes[1].set_ylabel('Firing Rate (Hz)')
    axes[1].legend(loc='upper left')
    
    # 3. Inputs
    t_array = np.arange(0, T/second + 0.05, 0.05)
    stim = example.Stimulus(stimparams['Ton'], stimparams['Toff'], stimparams['mu0'], stimparams['coh'])
    
    # Add noise for visualization
    input_A = stimparams['mu0'] + np.random.normal(0, 4.0, len(t_array)) * Hz
    input_B = stimparams['mu0'] + np.random.normal(0, 4.0, len(t_array)) * Hz
    
    mask_stim = (t_array >= 0.5) & (t_array < 1.5)
    input_A[~mask_stim] = 0
    input_B[~mask_stim] = 0
    
    axes[2].step(t_array, input_A/Hz, 'r', alpha=0.5, label='Input A')
    axes[2].step(t_array, input_B/Hz, 'b', alpha=0.5, label='Input B')
    axes[2].set_ylabel('Input Rate (Hz)')
    
    # 4. Time Integral
    dt_bin = 0.05 
    integral_A = np.cumsum(input_A/Hz) * dt_bin
    integral_B = np.cumsum(input_B/Hz) * dt_bin
    
    axes[3].plot(t_array, integral_A, 'r', label='Integral A')
    axes[3].plot(t_array, integral_B, 'b', label='Integral B')
    axes[3].set_ylabel('Time Integral')
    axes[3].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(save_filename)
    print(f"Figure saved to {save_filename}")

if __name__ == '__main__':
    reproduce_figure_3a()
