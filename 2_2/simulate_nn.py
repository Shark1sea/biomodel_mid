import numpy as np
import matplotlib.pyplot as plt

# Activation g(h) as defined: 0 if h<1, h-1 if 1<=h<=2, 1 if h>2

def g(h):
    h = np.array(h)
    out = np.zeros_like(h, dtype=float)
    mask1 = (h >= 1) & (h <= 2)
    mask2 = h > 2
    out[mask1] = h[mask1] - 1.0
    out[mask2] = 1.0
    return out


def simulate(K=3, T=10, w0=2.0, alpha=1.0, h0=None):
    if h0 is None:
        h = np.zeros((T+1, K), dtype=float)
    else:
        h = np.zeros((T+1, K), dtype=float)
        h[0, :] = h0

    # external input when t>0: h_ext[k] = (0.5)**(k+1) + 1 with k starting at 0
    h_ext = np.array([ (0.5)**(k+1) + 1.0 for k in range(K) ])

    for t in range(T):
        g_t = g(h[t])
        for k in range(K):
            inhibitory = np.sum(g_t) - g_t[k]
            h[t+1, k] = w0 * g_t[k] - alpha * inhibitory + (h_ext[k] if t>=0 else 0.0)
    return h


if __name__ == '__main__':
    K = 3
    T = 10
    # simulate two cases: (w0=2, alpha=1) and (w0=1, alpha=1)
    h_a = simulate(K=K, T=T, w0=2.0, alpha=1.0)
    h_b = simulate(K=K, T=T, w0=1.0, alpha=1.0)

    times = np.arange(T+1)

    plt.figure(figsize=(12,6))

    # Plot h for case A
    plt.subplot(1,2,1)
    for k in range(K):
        plt.plot(times, h_a[:,k], marker='o', label=f'h{k+1}')
    plt.title('Dynamics (w0=2, alpha=1)')
    plt.xlabel('t')
    plt.ylabel('h_k')
    plt.legend()
    plt.grid(True)

    # Plot h for case B
    plt.subplot(1,2,2)
    for k in range(K):
        plt.plot(times, h_b[:,k], marker='o', label=f'h{k+1}')
    plt.title('Dynamics (w0=1, alpha=1)')
    plt.xlabel('t')
    plt.ylabel('h_k')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    out = 'nn_dynamics.png'
    plt.savefig(out, dpi=150)
    print(f'Saved figure to {out}')
