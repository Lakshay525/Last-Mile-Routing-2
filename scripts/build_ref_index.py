"""Build the reference-route index for order=ref.
Default: from the TRAIN split only (no test route is ever a reference).
--all:   from all 6,112 training-folder routes, for the evaluation folder (no evaluation route is in the training folder)."""
import argparse, pickle
from lmr import load_routes, split, config
from lmr.zoneplan import build_reference_index

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--all", action="store_true"); a = ap.parse_args()
    routes = load_routes(6112, 0)
    tr = routes if a.all else split(routes)[0]
    idx = build_reference_index(tr)
    name = "ref_index_all.pkl" if a.all else "ref_index.pkl"
    config.CACHE_DIR.mkdir(exist_ok=True); pickle.dump(idx, open(config.CACHE_DIR / name, "wb"))
    print(name, "reference routes:", sum(len(v) for v in idx.values()), "from", len(idx), "stations")
