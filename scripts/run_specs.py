"""Run any list of method specs (see lmr/methods.py) on the held-out test routes, in parallel.

Example:
  python scripts/run_specs.py --out results/exp10.csv driver nn lns:i=1500 zlns:order=greedy:w=2 ortools:t=10
"""
import argparse, json, subprocess, time
from pathlib import Path
from lmr import load_routes, split
from lmr.runner import run_parallel

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--n-routes", type=int, default=1000, help="size of the sampled pool the split is taken from")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N routes of the chosen split")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--subset", default=None, choices=["high_delivered"], help="restrict to the literature test set (1,107 routes)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--zone-model", default=None, help="checkpoint of the zone-order GNN (needed for order=gnn)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-packages", action="store_true", help="skip service-time / time-window metrics")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tr, va, te = split(load_routes(a.n_routes, 0))
    routes = {"train": tr, "val": va, "test": te, "all": tr + va + te}[a.split]
    if a.subset:
        from lmr.literature import high_delivered_ids
        keep = high_delivered_ids(); routes = [r for r in routes if r.rid in keep]
    routes = routes[: a.limit]
    if not a.no_packages:                      # service times + time windows, used as evaluation metrics only
        from lmr.packages import attach_package_info
        attach_package_info(routes)
    t0 = time.time()
    df = run_parallel(routes, a.specs, a.workers, a.zone_model, a.seed)
    out = Path(a.out); out.parent.mkdir(exist_ok=True); df.to_csv(out, index=False)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump(dict(vars(a), commit=commit, date=time.strftime("%Y-%m-%d"), n_routes_evaluated=len(routes),
                   wall_seconds=round(time.time() - t0)), open(out.with_suffix(".json"), "w"), indent=2)
    print(df.groupby("method")[["gap_vs_driver", "zone_sw", "sim_to_driver", "zone_order_sim", "secs"]].mean().round(3))
