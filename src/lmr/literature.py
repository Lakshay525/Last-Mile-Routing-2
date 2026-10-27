"""Tools to compare with the published results on this dataset (Amazon Last Mile Routing Research Challenge).

* high_delivered_ids(): the 1,107 routes used as test set by the winning team's paper (Cook et al.):
  driver quality "High" and no DELIVERY_ATTEMPTED package (so no return visits).
* official_score(): the challenge's score for a proposed sequence, re-implemented from the public scoring code
  (MIT-CAVE/rc-cli, scoring/score.py): score = SD * ERP-per-edit. Lower is better, 0 = same order as the driver.
"""
import json, pickle
import numpy as np
from . import config
from .fastlns import jit


def high_delivered_ids():
    """IDs of routes rated High whose packages were all delivered (cached in .cache/high_delivered.json)."""
    cp = config.CACHE_DIR / "high_delivered.json"
    if cp.exists():
        return set(json.load(open(cp)))
    config.CACHE_DIR.mkdir(exist_ok=True)
    meta = json.load(open(config.BUILD / "route_data.json"))
    pk = json.load(open(config.BUILD / "package_data.json"))
    ids = []
    for rid, m in meta.items():
        if m.get("route_score") != "High": continue
        # "undelivered" = a delivery attempt that failed (the driver returns to the stop). A REJECTED package does not
        # count: this definition gives exactly the paper's 1,107 routes (requiring every package DELIVERED gives 1,102).
        if not any(p.get("scan_status") == "DELIVERY_ATTEMPTED" for stop in pk[rid].values() for p in stop.values()):
            ids.append(rid)
    json.dump(sorted(ids), open(cp, "w"))
    return set(ids)


# ---------------------------------------------------------------- official score
@jit
def _erp(a, s, nm, g):
    """Edit distance with real penalty, as in the official code. a, s: index arrays (full lists including the depot at
    both ends); nm: normalised matrix. Options per cell: match/substitute (cost nm[a_i][s_j], +1 edit if a_i != s_j),
    delete head of a (cost g, +1 edit), insert head of s (cost g, +1 edit). Ties prefer the first option."""
    n = len(a); m = len(s)
    cost = np.zeros((n + 1, m + 1)); cnt = np.zeros((n + 1, m + 1))
    for i in range(n - 1, -1, -1):
        cost[i, m] = g * (n - i); cnt[i, m] = n - i
    for j in range(m - 1, -1, -1):
        cost[n, j] = g * (m - j); cnt[n, j] = m - j
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            c1 = cost[i + 1, j + 1] + nm[a[i], s[j]]
            k1 = cnt[i + 1, j + 1] + (0.0 if a[i] == s[j] else 1.0)
            c2 = cost[i + 1, j] + g; k2 = cnt[i + 1, j] + 1.0
            c3 = cost[i, j + 1] + g; k3 = cnt[i, j + 1] + 1.0
            d = min(c1, min(c2, c3))
            if d == c1:
                cost[i, j] = c1; cnt[i, j] = k1
            elif d == c2:
                cost[i, j] = c2; cnt[i, j] = k2
            else:
                cost[i, j] = c3; cnt[i, j] = k3
    return cost[0, 0], cnt[0, 0]


def official_score(T, proposed, actual=None, g=1000.0):
    """Challenge score of `proposed` (list of stop indices, depot 0 first) against `actual` (default: the driver's
    order, i.e. 0..n after re-indexing). T: the route's travel-time matrix."""
    T = np.asarray(T, dtype=np.float64); n_all = len(T)
    actual = np.arange(n_all) if actual is None else np.asarray(actual)
    prop = np.asarray(proposed)
    # normalised matrix: z-score over ALL entries, then shift so the minimum is 0  (== T / std)
    z = (T - T.mean()) / T.std(); nm = z - z.min()
    # sequence deviation on the interior stops (depot at both ends removed)
    ai = actual[1:]; pi = prop[1:]; n = len(ai)
    pos = np.empty(n_all, dtype=np.int64); pos[ai] = np.arange(n)
    p = pos[pi]
    sd = (2.0 / (n * (n - 1))) * float((np.abs(np.diff(p)) - 1).sum()) if n > 1 else 0.0
    # edit distance on the full closed lists
    al = np.concatenate([actual, [actual[0]]]); sl = np.concatenate([prop, [prop[0]]])
    tot, edits = _erp(al.astype(np.int64), sl.astype(np.int64), nm, g)
    erp = tot / edits if edits > 0 else 0.0
    return sd * erp


def closed_time(T, p):
    """Travel time of the closed tour p[0] -> ... -> p[-1] -> p[0] (the literature's tour definition)."""
    p = np.asarray(p); return float(T[p[:-1], p[1:]].sum() + T[p[-1], p[0]])
