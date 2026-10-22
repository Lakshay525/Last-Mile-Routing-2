import hashlib, json, pickle, random
from dataclasses import dataclass
import numpy as np
from . import config

@dataclass
class Route:
    rid: str; ids: list; xy: np.ndarray; zone: np.ndarray; zone_names: list
    T: np.ndarray; actual: np.ndarray; score: str; station: str = None

def iter_travel_times(path, wanted=None):
    """Streams (route_id, {from:{to:t}}) without loading the 1.8 GB file."""
    import ijson
    with open(path, "rb") as f:
        for rid, mat in ijson.kvitems(f, ""):
            if wanted is None or rid in wanted:
                yield rid, mat

def build_route(rid, meta, seq, mat):
    """Index 0 = station; the driver's actual order becomes arange(n)."""
    stops = meta["stops"]
    ids = sorted(stops, key=(lambda s: seq["actual"][s]) if seq is not None
                 else (lambda s: stops[s]["type"] != "Station"))
    xy = np.array([[stops[s]["lat"], stops[s]["lng"]] for s in ids], dtype=float)
    zid = lambda s: stops[s]["zone_id"] if isinstance(stops[s]["zone_id"], str) else None  # station = NaN
    znames = sorted({zid(s) for s in ids if zid(s)}); zmap = {z: i for i, z in enumerate(znames)}
    zone = np.array([zmap.get(zid(s), -1) for s in ids])
    T = np.array([[float(mat[a][b]) for b in ids] for a in ids], dtype=np.float64)   # ijson gives Decimal
    actual = np.arange(len(ids)) if seq is not None else np.zeros(0, int)
    return Route(rid, ids, xy, zone, znames, T, actual, meta.get("route_score"), meta.get("station_code"))

def load_routes(n=50, seed=config.SEED):
    """Sample n training routes, cache them, return list[Route]."""
    config.CACHE_DIR.mkdir(exist_ok=True)
    cache = config.CACHE_DIR / f"routes_{n}_{seed}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    meta = json.load(open(config.BUILD / "route_data.json"))
    seqs = json.load(open(config.BUILD / "actual_sequences.json"))
    pick = set(random.Random(seed).sample(sorted(meta), n))
    routes = [build_route(rid, meta[rid], seqs[rid], mat)
              for rid, mat in iter_travel_times(config.BUILD / "travel_times.json", pick)]
    pickle.dump(routes, open(cache, "wb"))
    return routes

def split_of(rid):
    """Deterministic 60/20/20 split decided by the route id, so it never changes with n or seed."""
    h = int(hashlib.md5(rid.encode()).hexdigest(), 16) % 100
    return "train" if h < 60 else "val" if h < 80 else "test"

def split(routes):
    out = {"train": [], "val": [], "test": []}
    for r in routes: out[split_of(r.rid)].append(r)
    return out["train"], out["val"], out["test"]

def sub_route(r, nodes):
    """Sub-instance on `nodes` (sorted ascending keeps the driver's order = arange)."""
    return Route(r.rid, [r.ids[i] for i in nodes], r.xy[nodes], r.zone[nodes], r.zone_names,
                 r.T[np.ix_(nodes, nodes)], np.arange(len(nodes)), r.score, r.station)

def zone_subroutes(r):
    """One sub-instance per zone: entry = driver's previous stop, then the zone's stops."""
    out = []
    for z in sorted(set(r.zone[1:])):
        m = [i for i in range(1, len(r.zone)) if r.zone[i] == z]
        if len(m) >= 3: out.append(sub_route(r, [m[0] - 1] + m))
    return out