"""Compiled (numba) version of the LNS engine in pathlns.py. Same algorithm and parameters, much faster.

If numba is not installed, HAVE_NUMBA is False and pathlns.lns_path falls back to the pure-Python engine.
The random number generator differs from the Python engine, so results are comparable in distribution, not
run for run.
"""
import numpy as np

try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:  # numba missing or not supported on this Python
    HAVE_NUMBA = False

if HAVE_NUMBA:
    jit = nb.njit(cache=True)
else:
    def jit(f): return f


@jit
def path_cost_nb(C, p):
    s = 0.0
    for i in range(len(p) - 1):
        s += C[p[i], p[i + 1]]
    return s


@jit
def lateness_nb(T, p, svc, ts, te):
    """Total seconds of time-window lateness along path p (timeline of packages.time_metrics)."""
    t = 0.0; late = 0.0
    for i in range(len(p) - 1):
        a = p[i]; b = p[i + 1]
        t += T[a, b]
        if (not np.isnan(ts[b])) and t < ts[b]:
            t = ts[b]
        if (not np.isnan(te[b])) and t > te[b]:
            late += t - te[b]
        t += svc[b]
    return late


@jit
def lns_core(T, C, zone, init, iters, seed, fixed_end, use_tw, svc, ts, te, mu,
             kmin_div, kmax_div, temp_factor, final_temp):
    """Adaptive LNS. Destroy: 0 random, 1 related (Shaw), 2 worst detour, 3 zone. Repair: 0 greedy, 1 regret-2.
    Objective: sum C along the path (+ mu * lateness if use_tw); path[0] fixed, path[-1] fixed if fixed_end."""
    np.random.seed(seed)
    n = len(init)
    cur = init.copy()
    hi = n - 1 if fixed_end else n          # movable positions are 1 .. hi-1
    movable = hi - 1
    cc = path_cost_nb(C, cur)
    if use_tw:
        cc += mu * lateness_nb(T, cur, svc, ts, te)
    best = cur.copy(); bc = cc
    if movable < 2 or iters <= 0:
        return best, bc
    zone_ok = False
    for j in range(2, hi):
        if zone[cur[j]] != zone[cur[1]]:
            zone_ok = True
    wD = np.ones(4); wR = np.ones(2)
    kmin = max(2, movable // kmin_div); kmax = max(3, movable // kmax_div)
    temp = max(temp_factor * cc / n, 1e-3)
    cool = (final_temp / temp) ** (1.0 / iters)
    removed = np.zeros(n, np.bool_)
    keep = np.empty(n, np.int64); rem = np.empty(n, np.int64)
    posv = np.empty(n, np.int64); dv = np.empty(n)
    for it in range(iters):
        # ---- choose operators
        s = wD[0] + wD[1] + wD[2] + (wD[3] if zone_ok else 0.0)
        u = np.random.random() * s; d = 0; acc = wD[0]
        while d < 3 and u > acc:
            d += 1
            acc += wD[d] if (d < 3 or zone_ok) else 0.0
        if d == 3 and not zone_ok:
            d = 0
        r = 0 if np.random.random() * (wR[0] + wR[1]) < wR[0] else 1
        k = kmin + int(np.random.random() * (kmax - kmin + 1))
        if k > movable - 1:
            k = movable - 1
        # ---- destroy: mark k positions
        for j in range(n):
            removed[j] = False
        if d == 0:
            m = hi - 1
            for j in range(m):
                posv[j] = j + 1
            for j in range(k):
                q = j + int(np.random.random() * (m - j))
                tmp = posv[j]; posv[j] = posv[q]; posv[q] = tmp
                removed[posv[j]] = True
        elif d == 1:
            sp = 1 + int(np.random.random() * movable)
            sn = cur[sp]
            m = hi - 1
            for j in range(m):
                posv[j] = j + 1
                dv[j] = T[sn, cur[j + 1]] + T[cur[j + 1], sn]
            order = np.argsort(dv[:m])
            for j in range(k):
                removed[posv[order[j]]] = True
        elif d == 2:
            m = hi - 1
            for j in range(m):
                p = j + 1
                dt = C[cur[p - 1], cur[p]]
                if p + 1 < n:
                    dt += C[cur[p], cur[p + 1]] - C[cur[p - 1], cur[p + 1]]
                posv[j] = p
                dv[j] = -dt
            order = np.argsort(dv[:m])
            for j in range(k):
                removed[posv[order[j]]] = True
        else:
            sp = 1 + int(np.random.random() * movable)
            z = zone[cur[sp]]; cnt = 0
            for j in range(1, hi):
                if zone[cur[j]] == z and cnt < k:
                    removed[j] = True; cnt += 1
        m = 0; nr = 0
        for j in range(n):
            if removed[j]:
                rem[nr] = cur[j]; nr += 1
            else:
                keep[m] = cur[j]; m += 1
        # ---- repair
        while nr > 0:
            bx = -1; bpos = -1; bkey = -1e300; bc1 = 0.0
            nposn = m - 1 if fixed_end else m          # insertion after keep[j], j = 0 .. nposn-1
            for xi in range(nr):
                x = rem[xi]
                c1 = 1e300; c2 = 1e300; p1 = 0
                for j in range(nposn):
                    a = keep[j]
                    if j + 1 < m:
                        b = keep[j + 1]
                        c = C[a, x] + C[x, b] - C[a, b]
                    else:
                        c = C[a, x]
                    if c < c1:
                        c2 = c1; c1 = c; p1 = j
                    elif c < c2:
                        c2 = c
                if r == 1:
                    key = (c2 - c1) if c2 < 1e299 else 0.0
                else:
                    key = -c1
                if key > bkey:
                    bkey = key; bx = xi; bpos = p1
            x = rem[bx]
            for j in range(m, bpos + 1, -1):
                keep[j] = keep[j - 1]
            keep[bpos + 1] = x
            m += 1
            rem[bx] = rem[nr - 1]; nr -= 1
        qc = path_cost_nb(C, keep)
        if use_tw:
            qc += mu * lateness_nb(T, keep, svc, ts, te)
        reward = 0.0
        if qc < bc - 1e-9:
            best[:] = keep; bc = qc; reward = 5.0
        if qc < cc:
            reward = max(reward, 2.0)
        if qc < cc or np.random.random() < np.exp(max((cc - qc) / temp, -50.0)):
            cur[:] = keep; cc = qc
            reward = max(reward, 0.5)
        wD[d] = 0.9 * wD[d] + 0.1 * (reward + 0.05)
        wR[r] = 0.9 * wR[r] + 0.1 * (reward + 0.05)
        temp *= cool
    return best, bc


def lns_fast(T, C, zone, init, iters=800, seed=0, fixed_end=False, tw=None, kmin_div=40, kmax_div=8,
             temp_factor=0.02, final_temp=0.01):
    """Python wrapper. tw = (service, tw_start, tw_end, mu) or None. Returns (path list, cost, {})."""
    T = np.ascontiguousarray(T, dtype=np.float64); C = np.ascontiguousarray(C, dtype=np.float64)
    zone = np.ascontiguousarray(zone, dtype=np.int64); init = np.asarray(init, dtype=np.int64)
    if tw is None:
        svc = ts = te = np.zeros(1); mu = 0.0; use = False
    else:
        svc, ts, te, mu = (np.ascontiguousarray(tw[0], dtype=np.float64), np.ascontiguousarray(tw[1], dtype=np.float64),
                           np.ascontiguousarray(tw[2], dtype=np.float64), float(tw[3])); use = True
    best, bc = lns_core(T, C, zone, init, int(iters), int(seed), bool(fixed_end), use, svc, ts, te, mu,
                        int(kmin_div), int(kmax_div), float(temp_factor), float(final_temp))
    return [int(x) for x in best], float(bc), {}


@jit
def local_search_nb(C, path, fixed_end):
    """First-improvement local search to a local optimum: 2-opt (segment reversal) and Or-opt (move a segment of
    1-3 stops, either orientation). Exact for asymmetric C. path[0] (and path[-1] if fixed_end) never move."""
    p = path.copy(); n = len(p)
    hi = n - 1 if fixed_end else n                 # movable positions 1 .. hi-1
    F = np.zeros(n); B = np.zeros(n)
    improved = True
    while improved:
        improved = False
        for k in range(n - 1):
            F[k + 1] = F[k] + C[p[k], p[k + 1]]
            B[k + 1] = B[k] + C[p[k + 1], p[k]]
        # ---- 2-opt: reverse p[i..j]
        for i in range(1, hi - 1):
            for j in range(i + 1, hi):
                old = C[p[i - 1], p[i]] + (F[j] - F[i])
                new = C[p[i - 1], p[j]] + (B[j] - B[i])
                if j + 1 < n:
                    old += C[p[j], p[j + 1]]
                    new += C[p[i], p[j + 1]]
                if new < old - 1e-9:
                    a = i; b = j
                    while a < b:
                        t = p[a]; p[a] = p[b]; p[b] = t; a += 1; b -= 1
                    improved = True
                    for k in range(n - 1):
                        F[k + 1] = F[k] + C[p[k], p[k + 1]]
                        B[k + 1] = B[k] + C[p[k + 1], p[k]]
        # ---- Or-opt: move p[i..i+L-1] between p[j] and p[j+1] (or to the end), forward or reversed
        for L in range(1, 4):
            i = 1
            while i + L <= hi:
                s0 = p[i]; s1 = p[i + L - 1]
                prv = p[i - 1]
                has_next = i + L < n
                rm = C[prv, s0]
                if has_next:
                    nxt = p[i + L]; rm += C[s1, nxt] - C[prv, nxt]
                intf = F[i + L - 1] - F[i]; intr = B[i + L - 1] - B[i]
                best = -1e-9; bj = -1; brev = False
                jmax = n - 1 if not fixed_end else n - 2          # insert after p[j]; j = n-1 means append (free end only)
                for j in range(0, jmax + 1):
                    if j >= i - 1 and j <= i + L - 1:
                        continue
                    a = p[j]
                    if j + 1 < n:
                        b = p[j + 1]
                        base = C[a, b]
                        fwd = C[a, s0] + C[s1, b] - base
                        rev = C[a, s1] + C[s0, b] - base
                    else:
                        fwd = C[a, s0]; rev = C[a, s1]
                    d1 = fwd - rm
                    d2 = rev - rm + (intr - intf)
                    if d1 < best:
                        best = d1; bj = j; brev = False
                    if L > 1 and d2 < best:
                        best = d2; bj = j; brev = True
                if bj >= 0:
                    seg = p[i:i + L].copy()
                    if brev:
                        seg = seg[::-1].copy()
                    rest = np.empty(n - L, np.int64); m = 0
                    for q in range(n):
                        if q < i or q >= i + L:
                            rest[m] = p[q]; m += 1
                    pos = bj if bj < i else bj - L                 # index in rest of the node we insert after
                    out = np.empty(n, np.int64); m = 0
                    for q in range(pos + 1):
                        out[m] = rest[q]; m += 1
                    for q in range(L):
                        out[m] = seg[q]; m += 1
                    for q in range(pos + 1, n - L):
                        out[m] = rest[q]; m += 1
                    p[:] = out
                    improved = True
                    for k in range(n - 1):
                        F[k + 1] = F[k] + C[p[k], p[k + 1]]
                        B[k + 1] = B[k] + C[p[k + 1], p[k]]
                i += 1
    return p


def polish(C, path, fixed_end=False):
    C = np.ascontiguousarray(C, dtype=np.float64)
    return [int(x) for x in local_search_nb(C, np.asarray(path, dtype=np.int64), bool(fixed_end))]


# ====================================================================== simulated annealing (compiled)
@jit
def _edge_local(C, p, i, j, n):
    """Sum of the cost of the distinct edges touching positions i and j (edge e = p[e] -> p[e+1])."""
    s = 0.0
    e0 = i - 1; e1 = i; e2 = j - 1; e3 = j
    if e0 >= 0 and e0 + 1 < n:
        s += C[p[e0], p[e0 + 1]]
    if e1 + 1 < n:
        s += C[p[e1], p[e1 + 1]]
    if e2 != e1 and e2 >= 0 and e2 + 1 < n:
        s += C[p[e2], p[e2 + 1]]
    if e3 != e1 and e3 + 1 < n:
        s += C[p[e3], p[e3 + 1]]
    return s


@jit
def sa_core(C, init, iters, seed, fixed_end, t0f, tf, p2opt, poropt):
    """Simulated annealing on a path with the objective sum(C). Moves: 2-opt segment reversal (prob p2opt),
    Or-opt move of a segment of 1-3 stops (prob poropt), swap of two stops (the rest). Metropolis acceptance,
    geometric cooling from t0f * cost / n down to tf. path[0] (and path[-1] if fixed_end) never move."""
    np.random.seed(seed)
    n = len(init); p = init.copy()
    hi = n - 1 if fixed_end else n
    m = hi - 1
    cc = path_cost_nb(C, p)
    best = p.copy(); bc = cc
    if m < 2 or iters <= 0:
        return best, bc
    temp = max(t0f * cc / n, 1e-3)
    cool = (tf / temp) ** (1.0 / iters)
    out = np.empty(n, np.int64)
    for it in range(iters):
        u = np.random.random()
        delta = 0.0; kind = -1; ai = 0; aj = 0; aL = 0
        if u < p2opt:
            i = 1 + int(np.random.random() * m); j = 1 + int(np.random.random() * m)
            if i != j:
                if i > j:
                    t = i; i = j; j = t
                old = C[p[i - 1], p[i]]; new = C[p[i - 1], p[j]]
                for k in range(i, j):
                    old += C[p[k], p[k + 1]]; new += C[p[k + 1], p[k]]
                if j + 1 < n:
                    old += C[p[j], p[j + 1]]; new += C[p[i], p[j + 1]]
                delta = new - old; kind = 0; ai = i; aj = j
        elif u < p2opt + poropt:
            L = 1 + int(np.random.random() * 3); i = 1 + int(np.random.random() * m)
            if i + L - 1 <= hi - 1:
                qmax = n - 2 if fixed_end else n - 1
                q = int(np.random.random() * (qmax + 1))
                if q < i - 1 or q > i + L - 1:
                    s0 = p[i]; s1 = p[i + L - 1]; prv = p[i - 1]
                    rm = C[prv, s0]
                    if i + L < n:
                        nxt = p[i + L]; rm += C[s1, nxt] - C[prv, nxt]
                    a = p[q]
                    if q + 1 < n:
                        b = p[q + 1]; ins = C[a, s0] + C[s1, b] - C[a, b]
                    else:
                        ins = C[a, s0]
                    delta = ins - rm; kind = 1; ai = i; aj = q; aL = L
        else:
            i = 1 + int(np.random.random() * m); j = 1 + int(np.random.random() * m)
            if i != j:
                if i > j:
                    t = i; i = j; j = t
                before = _edge_local(C, p, i, j, n)
                t = p[i]; p[i] = p[j]; p[j] = t
                after = _edge_local(C, p, i, j, n)
                t = p[i]; p[i] = p[j]; p[j] = t           # undo; re-apply below if accepted
                delta = after - before; kind = 2; ai = i; aj = j
        if kind >= 0 and (delta < 0.0 or np.random.random() < np.exp(max(-delta / temp, -50.0))):
            if kind == 0:
                a_ = ai; b_ = aj
                while a_ < b_:
                    t = p[a_]; p[a_] = p[b_]; p[b_] = t; a_ += 1; b_ -= 1
            elif kind == 1:
                seg_start = ai; L = aL; q = aj
                mm = 0
                if q < seg_start:
                    for k in range(q + 1):
                        out[mm] = p[k]; mm += 1
                    for k in range(L):
                        out[mm] = p[seg_start + k]; mm += 1
                    for k in range(q + 1, seg_start):
                        out[mm] = p[k]; mm += 1
                    for k in range(seg_start + L, n):
                        out[mm] = p[k]; mm += 1
                else:
                    for k in range(seg_start):
                        out[mm] = p[k]; mm += 1
                    for k in range(seg_start + L, q + 1):
                        out[mm] = p[k]; mm += 1
                    for k in range(L):
                        out[mm] = p[seg_start + k]; mm += 1
                    for k in range(q + 1, n):
                        out[mm] = p[k]; mm += 1
                p[:] = out
            else:
                t = p[ai]; p[ai] = p[aj]; p[aj] = t
            cc += delta
            if cc < bc - 1e-9:
                best[:] = p; bc = cc
        temp *= cool
    return best, path_cost_nb(C, best)


def sa_fast(C, init, iters=500000, seed=0, fixed_end=False, t0f=0.05, tf=0.001, p2opt=0.4, poropt=0.4):
    """Returns (path list, cost under C, {})."""
    C = np.ascontiguousarray(C, dtype=np.float64); init = np.asarray(init, dtype=np.int64)
    best, bc = sa_core(C, init, int(iters), int(seed), bool(fixed_end), float(t0f), float(tf), float(p2opt), float(poropt))
    return [int(x) for x in best], float(bc), {}
