"""Build the reference-route index for order=ref from the TRAIN split only (no test route is ever a reference)."""
import pickle
from lmr import load_routes, split, config
from lmr.zoneplan import build_reference_index

if __name__ == "__main__":
    tr, _, _ = split(load_routes(6112, 0))
    idx = build_reference_index(tr)
    config.CACHE_DIR.mkdir(exist_ok=True); pickle.dump(idx, open(config.CACHE_DIR / "ref_index.pkl", "wb"))
    print("reference routes:", sum(len(v) for v in idx.values()), "from", len(idx), "stations")
