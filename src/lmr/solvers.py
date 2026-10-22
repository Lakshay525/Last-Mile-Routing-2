import math, random
import numpy as np
from .evaluate import cost
from . import config

def nearest_neighbour(T, start=0):
    n = len(T); seen = np.zeros(n, bool); seen[start] = True; p = [start]
    for _ in range(n - 1):
        d = T[p[-1]].copy(); d[seen] = np.inf
        j = int(d.argmin()); p.append(j); seen[j] = True
    return p

# ---------------- simulated annealing ----------------
def _neighbour(p, rng):               # station (position 0) is never moved
    n = len(p); q = p.copy(); m = rng.random()
    if m < 0.4:                       # 2-opt (cost recomputed since T is asymmetric)
        i, j = sorted(rng.sample(range(1, n), 2)); q[i:j+1] = q[i:j+1][::-1]
    elif m < 0.8:                     # or-opt: move 1-3 stops
        L = rng.randint(1, 3); i = rng.randint(1, max(1, n - L)); seg = q[i:i+L]; del q[i:i+L]
        k = rng.randint(1, len(q)); q[k:k] = seg
    else:                             # swap
        i, j = rng.sample(range(1, n), 2); q[i], q[j] = q[j], q[i]
    return q

def simulated_annealing(T, init=None, iters=30000, t0=None, alpha=None, seed=0, objective=None):
    rng = random.Random(seed); obj = objective or (lambda p: cost(T, p))
    cur = list(init) if init is not None else nearest_neighbour(T)
    cc = obj(cur); best, bc = cur, cc
    t0 = t0 or 0.05 * cc / len(cur); alpha = alpha or (1e-3 / t0) ** (1 / iters); t = t0
    for _ in range(iters):
        q = _neighbour(cur, rng); qc = obj(q)
        if qc < cc or rng.random() < math.exp((cc - qc) / t):
            cur, cc = q, qc
            if cc < bc: best, bc = cur, cc
        t *= alpha
    return best, bc

# ---------------- adaptive large neighbourhood search ----------------
def _destroy(p, T, zone, k, rng, kind):
    n = len(p); body = p[1:]
    if kind == "random":   rem = rng.sample(body, k)
    elif kind == "related":
        s = rng.choice(body); rem = sorted(body, key=lambda x: T[s, x] + T[x, s])[:k]
    elif kind == "zone":
        z = zone[rng.choice(body)]; rem = [x for x in body if zone[x] == z][:max(k, 1)]
    else:                                              # worst detour
        pos = {x: i for i, x in enumerate(p)}
        def detour(x):
            i = pos[x]; a = p[i-1]
            return T[a, x] + (T[x, p[i+1]] - T[a, p[i+1]] if i + 1 < n else 0)
        rem = sorted(body, key=detour, reverse=True)[:k]
    s = set(rem); return [x for x in p if x not in s], list(s)

def _insert_cost(T, p, x, i):
    a = p[i-1]
    return T[a, x] if i == len(p) else T[a, x] + T[x, p[i]] - T[a, p[i]]

def _repair(p, removed, T, kind):
    p = list(p); removed = list(removed)
    while removed:
        best = None
        for x in removed:
            cs = sorted((_insert_cost(T, p, x, i), i) for i in range(1, len(p) + 1))
            key = (cs[1][0] - cs[0][0]) if kind == "regret" else -cs[0][0]
            if best is None or key > best[0]: best = (key, x, cs[0][1])
        _, x, i = best; p.insert(i, x); removed.remove(x)
    return p

def lns(T, zone=None, init=None, iters=800, seed=0, adaptive=True, sa_accept=True):
    """Operator weights are learned online (a simple bandit 'agent')."""
    rng = random.Random(seed); n = len(T); zone = zone if zone is not None else np.zeros(n, int)
    cur = list(init) if init is not None else nearest_neighbour(T)
    cc = cost(T, cur); best, bc = cur, cc
    D = ["random", "related", "worst"] + (["zone"] if len(set(zone[1:])) > 1 else []); R = ["greedy", "regret"]
    wD = dict.fromkeys(D, 1.0); wR = dict.fromkeys(R, 1.0)
    temp = 0.02 * cc / n; cool = (1e-2 / temp) ** (1 / iters)
    for _ in range(iters):
        d = rng.choices(D, [wD[x] for x in D])[0]; r = rng.choices(R, [wR[x] for x in R])[0]
        k = min(rng.randint(max(2, n // 40), max(3, n // 8)), n - 2)
        part, rem = _destroy(cur, T, zone, k, rng, d); q = _repair(part, rem, T, r); qc = cost(T, q)
        reward = 0.0
        if qc < bc: best, bc, reward = q, qc, 5.0
        if qc < cc: reward = max(reward, 2.0)
        if qc < cc or (sa_accept and rng.random() < math.exp((cc - qc) / temp)):
            cur, cc = q, qc; reward = max(reward, 0.5)
        if adaptive:
            wD[d] = 0.9 * wD[d] + 0.1 * (reward + 0.05); wR[r] = 0.9 * wR[r] + 0.1 * (reward + 0.05)
        temp *= cool
    return best, bc, {"destroy": wD, "repair": wR}

# ---------------- zone-wise routing ----------------
def zone_wise(T, zone, solver, **kw):
    """Order zones greedily, solve each as a sub-problem (entry = last stop so far), concatenate."""
    zs = sorted(set(zone[1:])); members = {z: [i for i in range(1, len(zone)) if zone[i] == z] for z in zs}
    prev, order, left = [0], [], set(zs)
    while left:
        z = min(left, key=lambda z: T[np.ix_(prev, members[z])].mean())
        order.append(z); left.remove(z); prev = members[z]
    route = [0]
    for z in order:
        nodes = [route[-1]] + members[z]
        sp = solver(T[np.ix_(nodes, nodes)], **kw)[0] if len(nodes) > 2 else [0, 1]
        route += [nodes[i] for i in sp[1:]]
    return route

# ---------------- registry: name -> fn(route) -> sequence ----------------
SOLVERS = {
    "nn":         lambda r: nearest_neighbour(r.T),
    "driver":     lambda r: list(range(len(r.T))),
    "sa":         lambda r: simulated_annealing(r.T, iters=config.SA_ITERS)[0],
    "lns":        lambda r: lns(r.T, r.zone, iters=config.LNS_ITERS)[0],
    "lns_static": lambda r: lns(r.T, r.zone, iters=config.LNS_ITERS, adaptive=False)[0],   # ablation
    "zone_sa":    lambda r: zone_wise(r.T, r.zone, simulated_annealing, iters=config.ZONE_SA_ITERS),
    "zone_lns":   lambda r: zone_wise(r.T, r.zone, lambda T, **k: lns(T, iters=config.ZONE_LNS_ITERS)),
}