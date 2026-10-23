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
from .pathlns import penalised_matrix, lns_path, nearest_neighbour_path, path_cost


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
    path = lns_path(M, M, np.arange(Z + 1) - 1, init, iters=iters, seed=seed)[0]
    return [zs[i - 1] for i in path[1:]]


# ------------------------------------------------------------------ zone-specific LNS
def zone_window_lns(T, zone, order, lam=0.0, w=2, sweeps=1, init_iters=200, win_iters=300, seed=0):
    """Zone-specific LNS. w=1 -> windows are single zones (no overlap); w>=2 -> overlapping windows."""
    zone = np.asarray(zone); C = penalised_matrix(T, zone, lam); members = zone_members(zone)
    route = [0]; blocks = []                       # blocks[k] = (start, end) positions of zone order[k] in route
    for z in order:                                 # (2) solve each zone, entering from the previous stop
        nodes = [route[-1]] + members[z]; start = len(route)
        if len(nodes) > 2:
            sub = np.ix_(nodes, nodes)
            p = lns_path(T[sub], C[sub], zone[nodes], nearest_neighbour_path(C[sub]),
                         iters=init_iters, seed=seed)[0]
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
                         seed=seed + 1 + sw * Zn + k, fixed_end=fixed_end)[0]
            inner = [nodes[i] for i in p[1:(len(p) - 1 if fixed_end else len(p))]]
            route[lo:hi] = inner
    return route
