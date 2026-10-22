import time
import numpy as np, pandas as pd
from scipy.stats import kendalltau

def cost(T, p):
    """Open path from the station (index 0), no return leg, as in the challenge."""
    p = np.asarray(p); return T[p[:-1], p[1:]].sum()

def zone_switches(zone, p):
    z = zone[np.asarray(p)[1:]]; return int((z[1:] != z[:-1]).sum())

def similarity(p):
    """Kendall tau against the driver's order (actual = arange after re-indexing)."""
    pos = np.empty(len(p), int); pos[p] = np.arange(len(p))
    return kendalltau(pos, np.arange(len(p)))[0]

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