"""Method registry. A method is a SPEC STRING, so a table row says exactly how it was produced:

    driver                          the real route (reference)
    nn                              nearest neighbour
    lns:i=1500:l=0:mu=0             plain LNS, i iterations, zone-crossing penalty l (seconds per crossing),
                                    mu = penalty per second of time-window lateness (needs package data)
    zlns:order=gnn:w=2:l=0          zone-specific LNS; order in {greedy, driver, random, ztsp, gnn};
                                    w = zones per window (1 = no overlap); l = penalty;
                                    s = sweeps, ii = iterations per zone, wi = iterations per window
    ortools:t=10:l=0                OR-Tools guided local search, t seconds, same penalised objective
"""
import numpy as np
from .pathlns import plain_lns, penalised_matrix
from .zoneplan import order_greedy, order_driver, order_random, order_ztsp, zone_window_lns
from .solvers import nearest_neighbour

_CTX = {"zone_model": None, "zone_model_path": None}


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


def zone_order(r, name, seed=0):
    if name == "greedy": return order_greedy(r.T, r.zone)
    if name == "driver": return order_driver(r)
    if name == "random": return order_random(r.zone, seed)
    if name == "ztsp":   return order_ztsp(r.T, r.zone, seed=seed)
    if name == "gnn":
        from .zone_gnn import predict_order
        return predict_order(_CTX["zone_model"], r)
    raise ValueError(f"unknown zone order: {name}")


def run_spec(spec, r, seed=0):
    kind, p = parse(spec); lam = float(p.get("l", 0))
    if kind == "driver": return list(range(len(r.T)))
    if kind == "nn":     return nearest_neighbour(r.T)
    if kind == "lns":
        mu = float(p.get("mu", 0))
        if mu > 0 and getattr(r, "pkg", None) is None:
            raise ValueError("mu > 0 needs package data (run with package metrics enabled)")
        return plain_lns(r.T, r.zone, lam, int(p.get("i", 1500)), seed, pkg=getattr(r, "pkg", None), mu=mu)
    if kind == "zlns":
        order = zone_order(r, p.get("order", "greedy"), seed)
        return zone_window_lns(r.T, r.zone, order, lam, w=int(p.get("w", 2)), sweeps=int(p.get("s", 1)),
                               init_iters=int(p.get("ii", 200)), win_iters=int(p.get("wi", 300)), seed=seed)
    if kind == "ortools":
        from .baselines import ortools_route
        return ortools_route(penalised_matrix(r.T, r.zone, lam), seconds=float(p.get("t", 10)))
    raise ValueError(f"unknown method: {spec}")
