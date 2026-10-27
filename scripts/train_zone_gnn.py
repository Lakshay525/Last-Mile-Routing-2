"""Train the zone-order GNN by imitating the drivers' zone sequences, then compare it with simple rules on test routes."""
import argparse, json
from pathlib import Path
import numpy as np
from lmr import load_routes, split, config
from lmr.zoneplan import order_driver, order_greedy, order_random, order_ztsp, order_reference
from lmr.zone_gnn import train_zone_gnn, save_zone_model, predict_order, order_tau

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-routes", type=int, default=3000, help="size of the sampled pool (train/val parts are used)")
    ap.add_argument("--test-pool", type=int, default=1000, help="pool whose test split is the evaluation set")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--beam", type=int, default=20, help="beam width for the final test comparison")
    ap.add_argument("--ref", action="store_true", help="give the model the best reference route's zone order as features")
    ap.add_argument("--out", default="results/zone_gnn.pt")
    ap.add_argument("--eval-only", action="store_true", help="skip training: load --out and only run the test comparison")
    a = ap.parse_args()

    tr, va, _ = split(load_routes(a.n_routes, 0))
    _, _, te = split(load_routes(a.test_pool, 0))
    print(f"zone-order GNN: {len(tr)} train routes, {len(va)} val routes, {len(te)} test routes")
    ref = None
    if a.ref:
        import pickle
        ref = pickle.load(open(config.CACHE_DIR / "ref_index.pkl", "rb"))      # built by scripts/build_ref_index.py (train split only)
    import pickle
    REF = pickle.load(open(config.CACHE_DIR / "ref_index.pkl", "rb"))      # reference routes (train split only)
    if a.eval_only:
        from lmr.zone_gnn import load_zone_model
        pol, hist = load_zone_model(a.out), []
    else:
        pol, hist = train_zone_gnn(tr, va, epochs=a.epochs, seed=a.seed, d=a.d, layers=a.layers, ref=ref)
        Path(a.out).parent.mkdir(exist_ok=True); save_zone_model(pol, a.out)

    rules = {"gnn (greedy decoding)": lambda r: predict_order(pol, r),
             f"gnn (beam {a.beam})": lambda r: predict_order(pol, r, a.beam),
             "greedy (nearest zone next)": lambda r: order_greedy(r.T, r.zone),
             "ztsp (zone-level routing)": lambda r: order_ztsp(r.T, r.zone),
             "reference route (no learning)": lambda r: order_reference(r, REF),
             "random": lambda r: order_random(r.zone, 0)}
    res = {}
    for name, f in rules.items():
        taus = [order_tau(f(r), order_driver(r)) for r in te]
        res[name] = dict(mean_tau=float(np.mean(taus)), median_tau=float(np.median(taus)), std=float(np.std(taus)))
        print(f"test zone-order tau vs driver  {name:28s} mean {np.mean(taus):+.3f}  median {np.median(taus):+.3f}")
    json.dump(dict(history=hist, test=res, n_train=len(tr), n_val=len(va), n_test=len(te)),
              open(Path(a.out).with_suffix(".json"), "w"), indent=2)
