"""Method registry. A method is a SPEC STRING, so a table row says exactly how it was produced:

    driver                          the real route (reference)
    nn                              nearest neighbour
    sa:i=1000000:l=0:t0=0.05:tf=0.001  global simulated annealing (same objective and options as lns)
    zsa:order=..:w=3:ii=..:wi=..    zone-specific pipeline with SA instead of LNS inside zones/windows (the third approach)
    random:k=0                      random tour (k = trial number)
    lns:i=1500:l=0:mu=0             plain LNS (cl=1: closed tour back to the depot; extra keys: kmin, kmax, t0, tf, pol=1 for local-search polish), i iterations, zone-crossing penalty l (seconds per crossing),
                                    mu = penalty per second of time-window lateness (needs package data)
    zlns:order=gnn:w=2:l=0          zone-specific LNS; order in {greedy, driver, random, ztsp, ref, gnn};
                                    w = zones per window (1 = no overlap); l = penalty;
                                    s = sweeps, ii = iterations per zone, wi = iterations per window,
                                    beam = beam width for order=gnn (1 = greedy decoding),
                                    rho = penalty (s) per edge going backwards in the zone order,
                                    kmax = neighbourhood size, g = iterations of a final global LNS pass, pol=1 polish
    ortools:t=10:l=0                OR-Tools guided local search, t seconds, same penalised objective
"""
import numpy as np
from .pathlns import plain_lns, plain_sa, penalised_matrix
from .zoneplan import order_greedy, order_driver, order_random, order_ztsp, zone_window_lns, order_reference
from .solvers import nearest_neighbour

_CTX = {"zone_model": None, "zone_model_path": None, "ref_index": None}


def set_context(zone_model_path=None):
    """Per-process setup (called once in every worker): load the zone-order GNN if a path is given."""
    _CTX["zone_model_path"] = zone_model_path
    if zone_model_path:
        import torch
        torch.set_num_threads(1)
        from .zone_gnn import load_zone_model
        _CTX["zone_model"] = load_zone_model(zone_model_path)


def parse(spec):
    kind, *kv = spec.split(":"); params = {}
    for item in kv:
        k, v = item.split("="); params[k] = v
    return kind, params


def zone_order(r, name, seed=0, beam=1):
    if name == "greedy": return order_greedy(r.T, r.zone)
    if name == "driver": return order_driver(r)
    if name == "random": return order_random(r.zone, seed)
    if name == "ztsp":   return order_ztsp(r.T, r.zone, seed=seed)
    if name == "ref":
        if _CTX["ref_index"] is None:
            import pickle
            from . import config
            _CTX["ref_index"] = pickle.load(open(config.CACHE_DIR / "ref_index.pkl", "rb"))     # built by scripts/build_ref_index.py
        return order_reference(r, _CTX["ref_index"])
    if name == "gnn":
        from .zone_gnn import predict_order
        return predict_order(_CTX["zone_model"], r, beam)
    raise ValueError(f"unknown zone order: {name}")


def run_spec(spec, r, seed=0):
    kind, p = parse(spec); lam = float(p.get("l", 0))
    if kind == "driver": return list(range(len(r.T)))
    if kind == "nn":     return nearest_neighbour(r.T)
    if kind == "random":
        import random as _rnd
        rest = list(range(1, len(r.T))); _rnd.Random(seed * 1009 + int(p.get("k", 0))).shuffle(rest); return [0] + rest
    if kind == "lns":
        mu = float(p.get("mu", 0))
        if mu > 0 and getattr(r, "pkg", None) is None:
            raise ValueError("mu > 0 needs package data (run with package metrics enabled)")
        kw = {}
        if "kmin" in p: kw["kmin_div"] = int(p["kmin"])          # k_min = max(2, m // kmin)
        if "kmax" in p: kw["kmax_div"] = int(p["kmax"])          # k_max = max(3, m // kmax)
        if "t0" in p: kw["temp_factor"] = float(p["t0"])         # initial temperature = t0 * cost / n
        if "tf" in p: kw["final_temp"] = float(p["tf"])          # final temperature (seconds)
        return plain_lns(r.T, r.zone, lam, int(p.get("i", 1500)), seed, pkg=getattr(r, "pkg", None), mu=mu,
                         polish_final=p.get("pol", "0") == "1", closed=p.get("cl", "0") == "1", **kw)
    if kind == "sa":
        kw = {}
        if "t0" in p: kw["t0f"] = float(p["t0"])
        if "tf" in p: kw["tf"] = float(p["tf"])
        return plain_sa(r.T, r.zone, lam, int(p.get("i", 1000000)), seed, closed=p.get("cl", "0") == "1", **kw)
    if kind in ("zlns", "zsa"):
        order = zone_order(r, p.get("order", "greedy"), seed, int(p.get("beam", 1)))
        return zone_window_lns(r.T, r.zone, order, lam, w=int(p.get("w", 2)), sweeps=int(p.get("s", 1)),
                               init_iters=int(p.get("ii", 200)), win_iters=int(p.get("wi", 300)), seed=seed,
                               kmax_div=int(p["kmax"]) if "kmax" in p else None, global_iters=int(p.get("g", 0)),
                               solver="sa" if kind == "zsa" else "lns",
                               sa_kw={k2: float(p[k1]) for k1, k2 in (("t0", "t0f"), ("tf", "tf")) if k1 in p},
                               polish_final=p.get("pol", "0") == "1", rho=float(p.get("rho", 0)))
    if kind == "ortools":
        from .baselines import ortools_route
        return ortools_route(penalised_matrix(r.T, r.zone, lam), seconds=float(p.get("t", 10)))
    raise ValueError(f"unknown method: {spec}")
