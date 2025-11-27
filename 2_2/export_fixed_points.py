import csv
import math
from itertools import product

# Parameters (same as earlier script)
w0 = 2.0
alpha = 1.0
K = 3
h_ext = [ (0.5)**k + 1.0 for k in range(1, K+1) ]

states = {0: 'inactive', 1: 'linear', 2: 'saturated'}

results = []

for assignment in product([0,1,2], repeat=K):
    A = [i for i,s in enumerate(assignment) if s==1]
    S = [i for i,s in enumerate(assignment) if s==2]
    I = [i for i,s in enumerate(assignment) if s==0]
    n = len(A)
    m = len(S)

    d = 1.0 - w0 - alpha
    denom = d + n * alpha

    degenerate = False
    h = [None]*K
    G = [0.0]*K

    if n == 0:
        S_total = float(m)
        for k in range(K):
            if k in S:
                h[k] = (w0 + alpha) * 1.0 - alpha * S_total + h_ext[k]
                G[k] = 1.0
            else:
                h[k] = - alpha * S_total + h_ext[k]
                G[k] = 0.0
    else:
        if abs(denom) < 1e-12:
            degenerate = True
        else:
            H_A_ext = sum(h_ext[i] for i in A)
            S_total = (H_A_ext - n) / denom
            if abs(d) < 1e-12:
                degenerate = True
            else:
                for k in A:
                    xk = (h_ext[k] - 1.0 - alpha * S_total) / d
                    G[k] = xk
                    h[k] = xk + 1.0
                for k in S:
                    G[k] = 1.0
                    h[k] = (w0 + alpha) * 1.0 - alpha * S_total + h_ext[k]
                for k in I:
                    G[k] = 0.0
                    h[k] = - alpha * S_total + h_ext[k]

    self_consistent = False
    if not degenerate:
        ok = True
        for k in range(K):
            if assignment[k] == 0:
                if not (h[k] < 1.0 - 1e-9): ok = False
            elif assignment[k] == 1:
                if not (1.0 - 1e-9 <= h[k] <= 2.0 + 1e-9): ok = False
            elif assignment[k] == 2:
                if not (h[k] > 2.0 + 1e-9): ok = False
        self_consistent = ok

    stability_info = None
    if not degenerate and self_consistent:
        s = [1 if assignment[k]==1 else 0 for k in range(K)]
        n_active = sum(s)
        eigs = []
        eigs += [0.0] * (K - n_active)
        if n_active >= 1:
            if n_active - 1 > 0:
                eigs += [w0 + alpha] * (n_active - 1)
            eigs += [w0 - alpha * (n_active - 1)]
        stable = all(abs(ev) < 1.0 - 1e-12 for ev in eigs)
        stability_info = {'eigs': eigs, 'stable': stable, 's': s}

    results.append({
        'assignment': assignment,
        'A': A, 'S': S, 'I': I,
        'degenerate': degenerate,
        'self_consistent': self_consistent,
        'h': h,
        'G': G,
        'stability': stability_info
    })

# Filter self-consistent non-degenerate
found = [r for r in results if r['self_consistent'] and not r['degenerate']]

# Write CSV
csv_file = 'fixed_points.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['assignment', 'states', 'h1','h2','h3','G1','G2','G3','stable','eigs']
    writer.writerow(header)
    for r in found:
        assign = ''.join(str(x) for x in r['assignment'])
        states_list = [states[s] for s in r['assignment']]
        hvals = r['h']
        Gvals = r['G']
        stable = r['stability']['stable'] if r['stability'] else ''
        eigs = ';'.join(str(ev) for ev in (r['stability']['eigs'] if r['stability'] else []))
        row = [assign, '|'.join(states_list)] + [f'{v:.6g}' for v in hvals] + [f'{v:.6g}' for v in Gvals] + [stable, eigs]
        writer.writerow(row)

print(f'Wrote {len(found)} fixed points to {csv_file}')

# Attempt to plot (if matplotlib available)
try:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [ ''.join(str(x) for x in r['assignment']) for r in found ]
    h_values = np.array([ r['h'] for r in found ], dtype=float)
    stable_flags = [ r['stability']['stable'] for r in found ]

    x = np.arange(len(found))
    width = 0.25

    plt.figure(figsize=(10,4))
    for k in range(K):
        plt.bar(x + (k-1)*width, h_values[:,k], width, label=f'h{k+1}')

    # mark stability
    for i,stable in enumerate(stable_flags):
        plt.text(i, max(h_values[i])+0.1, 'S' if stable else 'U', ha='center', va='bottom', fontsize=9,
                 color='green' if stable else 'red')

    plt.xticks(x, labels)
    plt.ylabel('h*')
    plt.title('Fixed points h* values (assignment) — S=stable U=unstable')
    plt.legend()
    plt.tight_layout()
    out_png = 'fixed_points.png'
    plt.savefig(out_png, dpi=150)
    print(f'Saved plot to {out_png}')
except Exception as e:
    print('Matplotlib not available or plotting failed:', e)
    print('CSV still written; run the script in an environment with matplotlib to generate the plot.')
