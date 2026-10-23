"""Service times and time windows (from package_data.json), used as EVALUATION metrics only.

Per stop: service = sum of planned service seconds of its packages (independent of visiting order, so it
does not change which route is best; it only adds a constant to the route duration);
time window = the tightest window among its packages (latest start, earliest end), in seconds after departure.
Timeline for a route: leave the depot at t=0; arrive at b after travel; if early, wait until the window opens;
a stop is LATE if service would start after its window closes.
"""
import json, math, pickle
import datetime as dt
import numpy as np
from . import config

_FMT = "%Y-%m-%d %H:%M:%S"


def _cache_path():
    config.CACHE_DIR.mkdir(exist_ok=True); return config.CACHE_DIR / "pkginfo.pkl"


def package_info(routes):
    """{route id: dict(service, tw_start, tw_end)} aligned with Route.ids. Cached; loads the 359 MB file only
    if some route is missing from the cache."""
    cp = _cache_path(); cache = pickle.load(open(cp, "rb")) if cp.exists() else {}
    missing = [r for r in routes if r.rid not in cache]
    if missing:
        meta = json.load(open(config.BUILD / "route_data.json"))
        pk = json.load(open(config.BUILD / "package_data.json"))           # contains NaN literals; json accepts them
        for r in missing:
            m = meta[r.rid]; t0 = dt.datetime.strptime(m["date_YYYY_MM_DD"] + " " + m["departure_time_utc"], _FMT)
            n = len(r.ids); service = np.zeros(n); ts = np.full(n, np.nan); te = np.full(n, np.nan)
            for i, sid in enumerate(r.ids):
                pkgs = pk[r.rid].get(sid, {})
                for pdata in pkgs.values():
                    s = pdata.get("planned_service_time_seconds")
                    if s is not None and not math.isnan(s): service[i] += s
                    w = pdata.get("time_window", {})
                    a, b = w.get("start_time_utc"), w.get("end_time_utc")
                    if isinstance(a, str):
                        v = (dt.datetime.strptime(a, _FMT) - t0).total_seconds(); ts[i] = v if np.isnan(ts[i]) else max(ts[i], v)
                    if isinstance(b, str):
                        v = (dt.datetime.strptime(b, _FMT) - t0).total_seconds(); te[i] = v if np.isnan(te[i]) else min(te[i], v)
            cache[r.rid] = dict(service=service, tw_start=ts, tw_end=te)
        del pk
        pickle.dump(cache, open(cp, "wb"))
    return {r.rid: cache[r.rid] for r in routes}


def attach_package_info(routes):
    info = package_info(routes)
    for r in routes: r.pkg = info[r.rid]
    return routes


def time_metrics(r, p):
    """Simulate the route: returns duration (travel + service + waiting), service total, waiting, late stops, lateness."""
    info = getattr(r, "pkg", None)
    if info is None: return {}
    T, svc, ts, te = r.T, info["service"], info["tw_start"], info["tw_end"]
    t = 0.0; wait = 0.0; late = 0; late_s = 0.0
    for a, b in zip(p[:-1], p[1:]):
        t += T[a, b]
        if not np.isnan(ts[b]) and t < ts[b]: wait += ts[b] - t; t = ts[b]
        if not np.isnan(te[b]) and t > te[b]: late += 1; late_s += t - te[b]
        t += svc[b]
    return dict(service_total=float(svc.sum()), duration=t, wait_time=wait, late_stops=late, late_seconds=late_s,
                windowed_stops=int((~np.isnan(te)).sum()))
