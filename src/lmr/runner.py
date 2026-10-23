"""Parallel experiment runner (one task = one route x one method spec)."""
import time
import multiprocessing as mp
import numpy as np, pandas as pd
from . import methods
from .evaluate import route_metrics


def _init(zone_model_path):
    methods.set_context(zone_model_path)


def _task(args):
    r, spec, seed = args
    t = time.perf_counter(); p = methods.run_spec(spec, r, seed); dt = time.perf_counter() - t
    row = route_metrics(r, p); row.update(method=spec, secs=dt, seed=seed)
    return row


def run_parallel(routes, specs, workers=10, zone_model=None, seed=0, progress=True):
    tasks = [(r, s, seed) for r in routes for s in specs]
    rows, t0 = [], time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(zone_model,)) as pool:
        for k, row in enumerate(pool.imap_unordered(_task, tasks, chunksize=1), 1):
            rows.append(row)
            if progress and (k % max(1, len(tasks) // 10) == 0 or k == len(tasks)):
                print(f"  {k}/{len(tasks)} tasks  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows).sort_values(["route", "method"]).reset_index(drop=True)
