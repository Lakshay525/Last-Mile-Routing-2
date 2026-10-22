import argparse, json
from pathlib import Path
from lmr import load_routes, split
from lmr.data import zone_subroutes
from lmr.gnn import train, save_model

ap = argparse.ArgumentParser()
ap.add_argument("--n-routes", type=int, default=200)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--zone", action="store_true", help="train on per-zone sub-instances instead of whole routes")
ap.add_argument("--out", default="results/gnn.pt")
a = ap.parse_args()

tr, va, _ = split(load_routes(a.n_routes, a.seed))
if a.zone:
    tr = [s for r in tr for s in zone_subroutes(r)]; va = [s for r in va for s in zone_subroutes(r)]
pol, hist = train(tr, va, epochs=a.epochs)
Path(a.out).parent.mkdir(exist_ok=True)
save_model(pol, a.out); json.dump(hist, open(Path(a.out).with_suffix(".history.json"), "w"), indent=2)