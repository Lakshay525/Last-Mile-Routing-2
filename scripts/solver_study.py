"""What is inside OR-Tools? CP-SAT (circuit constraint) vs MIP (MTZ formulation, SCIP) vs our LNS.

Part A (small): first k stops of validation routes (k = 10..40) as small asymmetric open-path problems: can the exact
solvers prove optimality, how long do they take, and does LNS find the optimum?
Part B (full size): full routes: LNS (tuned) as a warm start for CP-SAT; how much does CP-SAT improve, and what is
the best PROVEN lower bound (so how far from optimal can LNS be)?

  python scripts/solver_study.py --out results/exp40_solver_study
"""
import argparse, time
import multiprocessing as mp
import numpy as np, pandas as pd
from lmr import load_routes, split
from lmr.pathlns import plain_lns, penalised_matrix, path_cost
from lmr.baselines import cpsat_route, mip_route


def _small(args):
    T, z, k, rid, secs = args
    Ts, zs = T[:k + 1, :k + 1], z[:k + 1]; C = penalised_matrix(Ts, zs, 0)
    row = dict(route=rid, k=k)
    t = time.time(); p = plain_lns(Ts, zs, 0, 3000, 0, kmax_div=2); row.update(lns_cost=path_cost(C, p), lns_secs=time.time() - t)
    t = time.time(); _, st, ob, bd = cpsat_route(C, secs, 1)
    row.update(cpsat_status=st, cpsat_cost=ob, cpsat_bound=bd, cpsat_secs=time.time() - t)
    t = time.time(); _, st, ob, bd = mip_route(C, secs)
    row.update(mip_status=st, mip_cost=ob, mip_bound=bd, mip_secs=time.time() - t)
    return row


def _full(args):
    T, z, rid, secs, workers = args
    C = penalised_matrix(T, z, 0)
    t = time.time(); p = plain_lns(T, z, 0, 5000, 0, kmax_div=2); lt = time.time() - t
    t = time.time(); _, st, ob, bd = cpsat_route(C, secs, workers, hint=p)
    return dict(route=rid, n=len(T), lns_cost=path_cost(C, p), lns_secs=lt, cpsat_status=st, cpsat_cost=ob, cpsat_bound=bd,
                cpsat_secs=time.time() - t)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/exp40_solver_study")
    ap.add_argument("--small-routes", type=int, default=15)
    ap.add_argument("--full-routes", type=int, default=20)
    ap.add_argument("--small-secs", type=float, default=20)
    ap.add_argument("--full-secs", type=float, default=60)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    _, va, _ = split(load_routes(3000, 0))
    with mp.get_context("spawn").Pool(a.procs) as pool:
        tasks = [(r.T, r.zone, k, r.rid, a.small_secs) for k in (10, 15, 20, 25, 30, 40) for r in va[: a.small_routes]]
        small = pd.DataFrame(pool.map(_small, tasks, chunksize=1)); small.to_csv(a.out + "_small.csv", index=False)
        print(small.groupby("k")[["lns_cost", "cpsat_cost", "mip_cost", "lns_secs", "cpsat_secs", "mip_secs"]].mean().round(2))
        print("proven optimal (of %d per size):" % a.small_routes)
        print(small.assign(cp=small.cpsat_status == "OPTIMAL", mi=small.mip_status == "OPTIMAL").groupby("k")[["cp", "mi"]].sum())
        tasks = [(r.T, r.zone, r.rid, a.full_secs, 2) for r in va[: a.full_routes]]
        full = pd.DataFrame(pool.map(_full, tasks, chunksize=1)); full.to_csv(a.out + "_full.csv", index=False)
        full["cpsat_improves_pct"] = 100 * (1 - full.cpsat_cost / full.lns_cost); full["proven_gap_pct"] = 100 * (full.cpsat_cost / full.cpsat_bound - 1)
        print(full[["n", "cpsat_improves_pct", "proven_gap_pct"]].describe().round(2))
