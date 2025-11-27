from itertools import product
import math

# Parameters (you can change w0, alpha, h_ext here)
w0 = 2.0
alpha = 1.0
K = 3
# external inputs for k=1..3: h_ext[k-1] = (0.5)**k + 1
h_ext = [ (0.5)**k + 1.0 for k in range(1, K+1) ]

states = {0: 'inactive', 1: 'linear', 2: 'saturated'}

results = []

for assignment in product([0,1,2], repeat=K):
    A = [i for i,s in enumerate(assignment) if s==1]
    S = [i for i,s in enumerate(assignment) if s==2]
    I = [i for i,s in enumerate(assignment) if s==0]
    n = len(A)
    m = len(S)

    # handle cases
    d = 1.0 - w0 - alpha  # d = 1 - (w0+alpha)
    denom = d + n * alpha

    degenerate = False
    h = [None]*K
    G = [0.0]*K

    if n == 0:
        # sum G = m (only saturated contribute)
        S_total = float(m)
        # compute h
        for k in range(K):
            if k in S:
                h[k] = (w0 + alpha) * 1.0 - alpha * S_total + h_ext[k]
                G[k] = 1.0
            else:
                h[k] = - alpha * S_total + h_ext[k]
                G[k] = 0.0
    else:
        # n >= 1
        if abs(denom) < 1e-12:
            # degenerate case: w0+alpha ~= 1 - n*alpha ??? treat as special
            degenerate = True
        else:
            H_A_ext = sum(h_ext[i] for i in A)
            S_total = (H_A_ext - n) / denom
            # compute G for linear A (x_k)
            if abs(d) < 1e-12:
                # d == 0 -> w0+alpha ==1, special degenerate
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

    # check self-consistency if not degenerate
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

    # linear stability analysis (analytic eigenvalues)
    stability_info = None
    if not degenerate and self_consistent:
        # s_k = 1 if linear
        s = [1 if assignment[k]==1 else 0 for k in range(K)]
        n_active = sum(s)
        eigs = []
        # zeros with multiplicity K-n_active
        eigs += [0.0] * (K - n_active)
        if n_active >= 1:
            if n_active - 1 > 0:
                eigs += [w0 + alpha] * (n_active - 1)
            eigs += [w0 - alpha * (n_active - 1)]
        # stability: all eigenvalues have abs < 1
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

# Print summary of found fixed points
print(f"Parameters: w0={w0}, alpha={alpha}, h_ext={h_ext}\n")
found = [r for r in results if r['self_consistent'] and not r['degenerate']]
if not found:
    print('No self-consistent non-degenerate fixed points found.')
else:
    print(f'Found {len(found)} self-consistent fixed points:\n')
    for idx,r in enumerate(found,1):
        assign = ''.join(str(x) for x in r['assignment'])
        print(f"#{idx}: assignment {assign}  (states: {[states[s] for s in r['assignment']]})")
        print("  h* = [" + ", ".join(f"{v:.6g}" for v in r['h']) + "]")
        print("  G  = [" + ", ".join(f"{v:.6g}" for v in r['G']) + "]")
        if r['stability']:
            eigs = r['stability']['eigs']
            print("  eigenvalues:", [f"{ev:.6g}" for ev in eigs])
            print("  linear stable:", r['stability']['stable'])
            print("  s (g' indicator):", r['stability']['s'])
        print()

# Also print degenerate/self-consistent info
deg = [r for r in results if r['degenerate']]
if deg:
    print('Degenerate cases (require special analysis):')
    for r in deg:
        print(' ', ''.join(str(x) for x in r['assignment']), 'states', [states[s] for s in r['assignment']])

# print all assignments that are self-consistent (including degenerate)
all_self = [r for r in results if r['self_consistent']]
print('\nAll self-consistent assignments (including degenerate):')
for r in all_self:
    print(' ', ''.join(str(x) for x in r['assignment']), [states[s] for s in r['assignment']])
