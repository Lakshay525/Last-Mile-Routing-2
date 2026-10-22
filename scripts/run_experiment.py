import argparse, json, subprocess, time
from pathlib import Path
from lmr import SOLVERS, load_routes, run_experiment, split

ap = argparse.ArgumentParser()
ap.add_argument("--methods", default="nn,sa,lns,lns_static,zone_sa,zone_lns")
ap.add_argument("--n-routes", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gnn", default=None, help="checkpoint, needed for gnn* methods")
ap.add_argument("--split", default="test", choices=["train", "val", "test"])
ap.add_argument("--out", required=True)
a = ap.parse_args()

pool = dict(SOLVERS)
if a.gnn:
    from lmr.gnn import load_model, gnn_solvers
    pool.update(gnn_solvers(load_model(a.gnn)))

tr, va, te = split(load_routes(a.n_routes, a.seed))
routes = {"train": tr, "val": va, "test": te}[a.split]
df = run_experiment(routes, {m: pool[m] for m in a.methods.split(",")})

out = Path(a.out); out.parent.mkdir(exist_ok=True); df.to_csv(out, index=False)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
json.dump(dict(vars(a), commit=commit, date=time.strftime("%Y-%m-%d"), n_routes_evaluated=len(routes)),
          open(out.with_suffix(".json"), "w"), indent=2)
print(df.groupby("method")[["gap_vs_driver", "zone_sw", "sim_to_driver", "secs"]].mean())