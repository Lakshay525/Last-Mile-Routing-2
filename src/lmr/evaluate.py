import time
import numpy as np, pandas as pd
from scipy.stats import kendalltau

def zone_order(zone, p):
    seen = []
    for i in p[1:]:
        if zone[i] not in seen: seen.append(zone[i])
    return seen

def zone_order_similarity(r, p):
    """Kendall tau between the order zones are first visited by p and by the driver."""
    drv, mine = zone_order(r.zone, r.actual), zone_order(r.zone, p)
    pos = {z: i for i, z in enumerate(mine)}
    return kendalltau(range(len(drv)), [pos[z] for z in drv])[0]

def cost(T, p):
    """Open path from the station (index 0), no return leg, as in the challenge."""
    p = np.asarray(p); return T[p[:-1], p[1:]].sum()

def zone_switches(zone, p):
    z = zone[np.asarray(p)[1:]]; return int((z[1:] != z[:-1]).sum())

def similarity(p):
    """Kendall tau against the driver's order (actual = arange after re-indexing)."""
    pos = np.empty(len(p), int); pos[p] = np.arange(len(p))
    return kendalltau(pos, np.arange(len(p)))[0]

def route_metrics(r, p):
    """All per-route metrics for a proposed visiting order p (list of stop indices, depot first).
    travel_time is in seconds from the travel-time matrix. zone_blocks_per_zone is (zone changes + 1) / zones:
    1.0 means every zone is visited in a single block. The penalised objective at any lambda is
    travel_time + lambda * zone_sw (up to a constant), so it can be computed afterwards from these columns."""
    p = list(map(int, p))
    assert sorted(p) == list(range(len(r.T))) and p[0] == 0, "invalid route"
    c = cost(r.T, p); base = cost(r.T, r.actual); nz = len(r.zone_names); zsw = zone_switches(r.zone, p)
    from .packages import time_metrics
    from .literature import official_score, closed_time
    return dict(route=r.rid, station=r.station, score=r.score, n=len(p), n_zones=nz,
                travel_time=c, driver_time=base, gap_vs_driver=c / base - 1,
                zone_sw=zsw, zone_blocks_per_zone=(zsw + 1) / nz,
                sim_to_driver=similarity(p), zone_order_sim=zone_order_similarity(r, p),
                closed_time=closed_time(r.T, p), driver_closed_time=closed_time(r.T, r.actual),
                official_score=official_score(r.T, p, r.actual), **time_metrics(r, p))


def run_experiment(routes, methods):
    """methods: {name: fn(route) -> sequence}. One row per (route, method)."""
    rows = []
    for r in routes:
        base = cost(r.T, r.actual)
        for name, f in methods.items():
            t = time.time(); p = f(r); dt = time.time() - t
            assert sorted(p) == list(range(len(r.T))) and p[0] == 0, f"{name} returned an invalid tour"
            c = cost(r.T, p)
            rows.append(dict(route=r.rid, n=len(p), station=r.station, score=r.score, method=name,
                             cost=c, gap_vs_driver=c / base - 1, zone_sw=zone_switches(r.zone, p),
                             sim_to_driver=similarity(p), secs=dt))
    return pd.DataFrame(rows)