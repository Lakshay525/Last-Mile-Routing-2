"""Zone-specific LNS (the 'first approach').

Pipeline
  1. a zone ORDER (rule-based, learned by a GNN, or the driver's own = oracle)
  2. solve each zone as a small path problem, entering from the stop where the previous zone ended
  3. re-optimise OVERLAPPING WINDOWS of w consecutive zones (stride 1 => each zone is in w windows),
     with the stop before the window and the stop after it held fixed. With a zone-crossing penalty
     lam > 0 a stop may move to a neighbouring zone only if the travel-time saving exceeds the penalty.
"""
import random
import numpy as np
from .pathlns import penalised_matrix, lns_path as _lns_path, sa_path, nearest_neighbour_path, path_cost, engine


# ------------------------------------------------------------------ zone structure
def zone_members(zone):
    """{zone id: [stop indices]} for all stops except the depot (index 0)."""
    out = {}
    for i in range(1, len(zone)):
        out.setdefault(int(zone[i]), []).append(i)
    return out


def zone_matrix(T, members, zs):
    """Asymmetric zone-to-zone travel time: mean of T[i, j] over i in zone a, j in zone b.
    Node 0 is the depot, nodes 1..Z follow the order of `zs`."""
    groups = [[0]] + [members[z] for z in zs]
    Z = len(groups); M = np.zeros((Z, Z))
    for a in range(Z):
        for b in range(Z):
            if a != b: M[a, b] = T[np.ix_(groups[a], groups[b])].mean()
    return M


# ------------------------------------------------------------------ zone orders
def order_driver(r):
    """Oracle: the order in which the driver first entered each zone."""
    seen = []
    for i in r.actual[1:]:
        z = int(r.zone[i])
        if z not in seen: seen.append(z)
    return seen


def order_greedy(T, zone):
    """Nearest zone next (mean travel time from the stops of the zone we are leaving)."""
    members = zone_members(zone); left = set(members); prev = [0]; order = []
    while left:
        z = min(left, key=lambda z: T[np.ix_(prev, members[z])].mean())
        order.append(z); left.remove(z); prev = members[z]
    return order


def order_random(zone, seed=0):
    zs = sorted(zone_members(zone)); random.Random(seed).shuffle(zs); return zs


def order_ztsp(T, zone, iters=300, seed=0):
    """Zone order from a small routing problem over zones (Tz = mean travel time between zones), solved by LNS."""
    members = zone_members(zone); zs = sorted(members)
    M = zone_matrix(T, members, zs); Z = len(zs)
    if Z < 3: return zs
    init = nearest_neighbour_path(M)
    path = _lns_path(M, M, np.arange(Z + 1) - 1, init, iters=iters, seed=seed)[0]
    return [zs[i - 1] for i in path[1:]]


# ------------------------------------------------------------------ zone-specific LNS
def zone_window_lns(T, zone, order, lam=0.0, w=2, sweeps=1, init_iters=200, win_iters=300, seed=0,
                    kmax_div=None, global_iters=0, polish_final=False, rho=0.0, solver="lns", sa_kw=None):
    """Zone-specific LNS. w=1 -> windows are single zones (no overlap); w>=2 -> overlapping windows.
    kmax_div: neighbourhood size for the compiled engine (k_max = m // kmax_div); global_iters > 0 adds a final
    LNS pass over the whole route starting from the zone-window solution; polish_final adds 2-opt/Or-opt.
    rho > 0 adds a penalty of rho seconds for every edge that goes BACKWARDS in the zone order (from a zone to a zone
    that comes earlier in `order`), a soft version of 'visit the zones in this order'."""
    kw = {"kmax_div": kmax_div} if (kmax_div is not None and engine() == "numba") else {}
    sa_kw = sa_kw or {}
    if solver == "sa":      # same pipeline, simulated annealing as the optimiser inside zones / windows (iterations = SA iterations)
        def lns_path(Tm, Cm, zm, init, iters=0, seed=0, fixed_end=False, **_):
            return sa_path(Cm, init, iters, seed, fixed_end, **sa_kw)
    else:
        lns_path = _lns_path
    zone = np.asarray(zone); C = penalised_matrix(T, zone, lam); members = zone_members(zone)
    if rho > 0:
        rank = np.full(len(zone), -1); rank[[i for z in order for i in members[z]]] = np.repeat(np.arange(len(order)), [len(members[z]) for z in order])
        C = C + float(rho) * (rank[:, None] > rank[None, :])
    route = [0]; blocks = []                       # blocks[k] = (start, end) positions of zone order[k] in route
    for z in order:                                 # (2) solve each zone, entering from the previous stop
        nodes = [route[-1]] + members[z]; start = len(route)
        if len(nodes) > 2:
            sub = np.ix_(nodes, nodes)
            p = lns_path(T[sub], C[sub], zone[nodes], nearest_neighbour_path(C[sub]),
                         iters=init_iters, seed=seed, **kw)[0]
            route += [nodes[i] for i in p[1:]]
        else:
            route += nodes[1:]
        blocks.append((start, len(route)))
    Zn = len(order); w = max(1, min(w, Zn))
    for sw in range(sweeps):                        # (3) overlapping windows over the zone-ordered route
        for k in range(0, Zn - w + 1):
            lo, hi = blocks[k][0], blocks[k + w - 1][1]
            if hi - lo < 2: continue
            fixed_end = hi < len(route)
            nodes = [route[lo - 1]] + route[lo:hi] + ([route[hi]] if fixed_end else [])
            sub = np.ix_(nodes, nodes)
            p = lns_path(T[sub], C[sub], zone[nodes], list(range(len(nodes))), iters=win_iters,
                         seed=seed + 1 + sw * Zn + k, fixed_end=fixed_end, **kw)[0]
            inner = [nodes[i] for i in p[1:(len(p) - 1 if fixed_end else len(p))]]
            route[lo:hi] = inner
    if global_iters > 0:                          # (4) global refinement from the zone-structured solution
        route = lns_path(T, C, zone, route, iters=global_iters, seed=seed + 7919, **kw)[0]
    if polish_final and engine() == "numba":
        from .fastlns import polish
        route = polish(C, route)
    return route


# ------------------------------------------------------------------ zone order from a reference training route
# Idea from the challenge winners' paper (Cook et al., section 4.5): for a new route R, pick the training route Q from
# the same depot that shares the most zones with R (weighted by driver quality: High 2, Medium 1.5, Low 1) and follow
# the order in which Q's driver visited those zones. Zones of R that Q never visited are inserted greedily.
_QW = {"High": 2.0, "Medium": 1.5, "Low": 1.0}


def build_reference_index(train_routes):
    """station -> list of (route id, quality weight, zone names in the order the driver first entered them)."""
    idx = {}
    for q in train_routes:
        names = [q.zone_names[z] for z in order_driver(q)]
        idx.setdefault(q.station, []).append((q.rid, _QW.get(q.score, 1.0), names))
    return idx


def best_reference_names(r, index):
    """Zone names (in driver order) of the best reference training route for r, or None. Never the route itself."""
    mine = {r.zone_names[z] for z in zone_members(r.zone)}
    best, bs = None, -1.0
    for rid, w, names in index.get(r.station, []):
        if rid == r.rid: continue
        s = w * len(mine & set(names))
        if s > bs: best, bs = names, s
    return best


def order_reference(r, index, T=None):
    """Zone order (list of the route's own zone ids) predicted from the best reference training route."""
    T = r.T if T is None else T
    mine = {r.zone_names[z]: z for z in zone_members(r.zone)}
    best = best_reference_names(r, index)
    seq = [mine[n] for n in best if n in mine] if best else []
    missing = [z for z in zone_members(r.zone) if z not in seq]
    if missing:                                                    # insert unseen zones where they add the least zone-to-zone time
        members = zone_members(r.zone); zs = sorted(members); M = zone_matrix(T, members, zs); pos = {z: i + 1 for i, z in enumerate(zs)}
        for z in sorted(missing, key=lambda z: -len(members[z])):
            bestc, bi = None, 0
            for i in range(len(seq) + 1):
                a = pos[seq[i - 1]] if i > 0 else 0
                c = M[a, pos[z]] + (M[pos[z], pos[seq[i]]] - M[a, pos[seq[i]]] if i < len(seq) else 0)
                if bestc is None or c < bestc: bestc, bi = c, i
            seq.insert(bi, z)
    return seq
