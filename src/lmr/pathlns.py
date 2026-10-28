"""LNS engine used by both approaches (plain LNS and zone-window LNS).

Objective of a path p:   J(p) = sum_i T[p_i, p_{i+1}]  +  lam * #{i : zone(p_i) != zone(p_{i+1})}
which is just sum_i C[p_i, p_{i+1}] with the penalised matrix C = T + lam * [zone differs].
The first node of the path is always fixed (depot, or the stop the driver is already at);
the last node can optionally be fixed too (needed when re-optimising a segment inside a longer route).
"""
import math, random
import numpy as np


def penalised_matrix(T, zone, lam, levels=None):
    """C = T + lam * 1[zone(i) != zone(j)] (+ w * 1[label(i) != label(j)] for every (label, w) in `levels`, the coarser
    clusters of hierarchy.py).  The depot (label -1) differs from every stop, which adds the same constant to every
    route, so it does not affect any comparison."""
    z = np.asarray(zone)
    C = np.asarray(T, dtype=np.float64) + float(lam) * (z[:, None] != z[None, :])
    for lab, w in (levels or []):
        lab = np.asarray(lab); C = C + float(w) * (lab[:, None] != lab[None, :])
    return C


def path_cost(C, p):
    p = np.asarray(p)
    return float(C[p[:-1], p[1:]].sum())


def nearest_neighbour_path(C, start=0, nodes=None):
    nodes = list(range(len(C))) if nodes is None else list(nodes)
    left = [x for x in nodes if x != start]; p = [start]
    while left:
        j = min(left, key=lambda x: C[p[-1], x]); p.append(j); left.remove(j)
    return p


# ------------------------------------------------------------------ destroy / repair
def _destroy(p, T, C, zone, k, rng, kind, fixed_end):
    """p: np.array path. Removes k movable nodes (never p[0], and never p[-1] when fixed_end)."""
    hi = len(p) - 1 if fixed_end else len(p)
    body = p[1:hi]
    if kind == "random":
        rem = rng.sample(list(body), k)
    elif kind == "related":                      # Shaw removal: nodes closest (travel time) to a random seed
        s = body[rng.randrange(len(body))]
        d = T[s, body] + T[body, s]
        rem = list(body[np.argsort(d)[:k]])
    elif kind == "zone":                         # pull out (part of) one zone
        z = zone[body[rng.randrange(len(body))]]
        rem = [x for x in body if zone[x] == z][:max(k, 1)]
    else:                                        # worst: largest detour contribution
        a, b = p[:-1], p[1:]
        # contribution of node p[i] (i>=1): C[p[i-1],p[i]] + C[p[i],p[i+1]] - C[p[i-1],p[i+1]]
        det = np.empty(len(p) - 1)
        det[:] = C[p[:-1], p[1:]]                # edge into p[i], i = 1..
        if len(p) > 2:
            det[:-1] += C[p[1:-1], p[2:]] - C[p[:-2], p[2:]]
        cand = np.arange(1, hi)                  # indices into p of movable nodes
        order = cand[np.argsort(-det[cand - 1])][:k]
        rem = list(p[order])
    s = set(int(x) for x in rem)
    keep = np.array([x for x in p if int(x) not in s], dtype=np.int64)
    return keep, [int(x) for x in rem]


def _repair(p, removed, C, kind, fixed_end):
    """Re-insert removed nodes, one at a time, at the cheapest (greedy) or highest-regret position."""
    p = np.asarray(p, dtype=np.int64); rem = list(removed)
    while rem:
        best = None
        pa, pb = p[:-1], p[1:]
        base = C[pa, pb]
        for x in rem:
            cs = C[pa, x] + C[x, pb] - base                 # insert between p[j] and p[j+1] -> index j+1
            if not fixed_end:
                cs = np.append(cs, C[p[-1], x])             # or append at the end
            j = int(np.argmin(cs)); c0 = cs[j]
            if kind == "regret":
                key = (np.partition(cs, 1)[1] - c0) if len(cs) > 1 else 0.0
            else:
                key = -c0
            if best is None or key > best[0]: best = (key, x, j + 1)
        _, x, idx = best
        p = np.insert(p, idx, x); rem.remove(x)
        pa, pb = p[:-1], p[1:]
    return p


def engine():
    """'numba' (compiled, default when numba is installed) or 'py' (pure Python; force with LMR_ENGINE=py)."""
    import os
    from .fastlns import HAVE_NUMBA
    return "numba" if HAVE_NUMBA and os.environ.get("LMR_ENGINE", "numba") != "py" else "py"


def lns_path(T, C, zone, init, iters=800, seed=0, fixed_end=False, adaptive=True, sa_accept=True, extra=None, tw=None, **kw):
    """Dispatcher: compiled engine if available, else the pure-Python one. tw = (service, tw_start, tw_end, mu)
    is the lateness penalty for the compiled engine; `extra` (a Python callable) is only used by the Python engine."""
    if engine() == "numba" and extra is None and adaptive and sa_accept:
        from .fastlns import lns_fast
        return lns_fast(T, C, zone, init, iters, seed, fixed_end, tw, **kw)
    if extra is None and tw is not None:
        extra = lateness_penalty(T, dict(service=np.asarray(tw[0]), tw_start=np.asarray(tw[1]), tw_end=np.asarray(tw[2])), tw[3])
    return lns_path_py(T, C, zone, init, iters, seed, fixed_end, adaptive, sa_accept, extra)


def lns_path_py(T, C, zone, init, iters=800, seed=0, fixed_end=False, adaptive=True, sa_accept=True, extra=None):
    """Adaptive LNS on a path (pure Python). Returns (best_path as list, best_cost under C, operator weights).
    T is used only to measure how 'related' two nodes are; the objective is C (+ extra(path) if given,
    e.g. a lateness penalty; it is used only to accept or reject a move, not to choose insertion positions)."""
    rng = random.Random(seed)
    p = np.asarray(init, dtype=np.int64); n = len(p)
    movable = n - 1 - (1 if fixed_end else 0)
    cc = path_cost(C, p) + (extra(p) if extra else 0.0)
    if movable < 2 or iters <= 0:
        return list(map(int, p)), cc, {}
    zone = np.asarray(zone)
    cur, best, bc = p, p, cc
    D = ["random", "related", "worst"] + (["zone"] if len(set(zone[p[1:]])) > 1 else [])
    R = ["greedy", "regret"]
    wD = dict.fromkeys(D, 1.0); wR = dict.fromkeys(R, 1.0)
    temp = max(0.02 * cc / n, 1e-3); cool = (1e-2 / temp) ** (1 / iters)
    kmin, kmax = max(2, movable // 40), max(3, movable // 8)
    for _ in range(iters):
        d = rng.choices(D, [wD[x] for x in D])[0]; r = rng.choices(R, [wR[x] for x in R])[0]
        k = min(rng.randint(kmin, kmax), movable - 1)
        part, rem = _destroy(cur, T, C, zone, k, rng, d, fixed_end)
        q = _repair(part, rem, C, r, fixed_end); qc = path_cost(C, q) + (extra(q) if extra else 0.0)
        reward = 0.0
        if qc < bc - 1e-9: best, bc, reward = q, qc, 5.0
        if qc < cc: reward = max(reward, 2.0)
        if qc < cc or (sa_accept and rng.random() < math.exp(max((cc - qc) / temp, -50.0))):
            cur, cc = q, qc; reward = max(reward, 0.5)
        if adaptive:
            wD[d] = 0.9 * wD[d] + 0.1 * (reward + 0.05); wR[r] = 0.9 * wR[r] + 0.1 * (reward + 0.05)
        temp *= cool
    return list(map(int, best)), bc, {"destroy": wD, "repair": wR}


def lateness_penalty(T, pkg, mu):
    """extra(path) = mu * total seconds of time-window lateness along the path (same timeline as packages.time_metrics)."""
    Tl = np.asarray(T).tolist(); svc = pkg["service"].tolist(); ts = pkg["tw_start"].tolist(); te = pkg["tw_end"].tolist()
    def extra(p):
        t = 0.0; late = 0.0; p = p.tolist()
        for a, b in zip(p[:-1], p[1:]):
            t += Tl[a][b]
            if ts[b] == ts[b] and t < ts[b]: t = ts[b]          # x == x is False for NaN (no window)
            if te[b] == te[b] and t > te[b]: late += t - te[b]
            t += svc[b]
        return mu * late
    return extra


def plain_lns(T, zone, lam=0.0, iters=1500, seed=0, init=None, pkg=None, mu=0.0, polish_final=False, closed=False, levels=None, **kw):
    """Plain LNS on the whole route (the 'second approach'). Returns the route as a list.
    mu > 0 (needs pkg) adds a penalty of mu per second of time-window lateness when accepting moves."""
    if closed:                                    # closed tour: depot -> stops -> back to the depot (literature's definition)
        n = len(T); T2 = np.zeros((n + 1, n + 1)); T2[:n, :n] = T; T2[:n, n] = np.asarray(T)[:, 0]; T2[n, :n] = np.asarray(T)[0]
        z2 = np.append(np.asarray(zone), -1)
        C2 = penalised_matrix(T2, z2, lam); start = nearest_neighbour_path(C2[:n, :n]) if init is None else list(init)
        path = lns_path(T2, C2, z2, start + [n], iters=iters, seed=seed, fixed_end=True, **kw)[0]
        return path[:-1]
    C = penalised_matrix(T, zone, lam, levels)
    init = nearest_neighbour_path(C) if init is None else list(init)
    tw = (pkg["service"], pkg["tw_start"], pkg["tw_end"], mu) if mu > 0 else None
    route = lns_path(T, C, zone, init, iters=iters, seed=seed, tw=tw, **kw)[0]
    if polish_final and engine() == "numba" and tw is None:      # local search ignores the lateness term
        from .fastlns import polish
        route = polish(C, route)
    return route


def sa_path(C, init, iters=500000, seed=0, fixed_end=False, **kw):
    """Simulated annealing on a path (compiled engine in fastlns.sa_core; needs numba for useful speed)."""
    from .fastlns import sa_fast
    return sa_fast(C, init, iters, seed, fixed_end, **kw)


def plain_sa(T, zone, lam=0.0, iters=1000000, seed=0, closed=False, **kw):
    """Global simulated annealing on the whole route (same objective, cost matrix and tours as plain_lns)."""
    if closed:
        n = len(T); T2 = np.zeros((n + 1, n + 1)); T2[:n, :n] = T; T2[:n, n] = np.asarray(T)[:, 0]; T2[n, :n] = np.asarray(T)[0]
        z2 = np.append(np.asarray(zone), -1); C2 = penalised_matrix(T2, z2, lam)
        return sa_path(C2, nearest_neighbour_path(C2[:n, :n]) + [n], iters, seed, True, **kw)[0][:-1]
    C = penalised_matrix(T, zone, lam)
    return sa_path(C, nearest_neighbour_path(C), iters, seed, False, **kw)[0]
