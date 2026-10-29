"""Run the final methods on the routes of the challenge's EVALUATION folder (3,052 routes), in resumable chunks.

  python scripts/run_eval.py --n 300 --seed 0 --out results/eval_pilot_s0.csv          # pilot on a random subset of 300 routes
  python scripts/run_eval.py --seed 0 --out results/eval_full_s0.csv                   # all routes, our methods
  python scripts/run_eval.py --ortools --out results/eval_full_ortools.csv             # OR-Tools (time-limited, run with the machine idle)

Settings are the ones fixed on validation routes (run_final.sh), no tuning on evaluation routes. The past-routes zone order uses all 6,112
training-folder routes (python scripts/build_ref_index.py --all), set via LMR_REF_INDEX.
Results are appended chunk by chunk, so an interrupted run continues where it stopped.
"""
import argparse, json, os, random, time
from pathlib import Path
import pandas as pd
from lmr import config
from lmr.data import load_eval_routes
from lmr.runner import run_parallel

LNS = "lns:i=5000:kmax=2"; ZL = "w=3:ii=1000:wi=1500:kmax=2"; ZS = "w=3:ii=50000:wi=300000:t0=0.5"


def specs(ortools):
    if ortools:
        return ["ortools:t=10", "ortools:t=10:l=60"]
    out = ["driver"]
    for L in (0, 60):
        out += [f"{LNS}:l={L}", f"zlns:order=ztsp:{ZL}:l={L}", f"zlns:order=ref:{ZL}:l={L}", f"zsa:order=ztsp:{ZS}:l={L}", f"zsa:order=ref:{ZS}:l={L}"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="random subset of this many routes (pilot). Default: all routes")
    ap.add_argument("--subset-seed", type=int, default=0, help="seed of the random subset (fixed, so the pilot is the same every time)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the optimisers")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--ortools", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("LMR_REF_INDEX", str(config.CACHE_DIR / "ref_index_all.pkl"))

    routes = load_eval_routes()
    if a.n:
        keep = set(random.Random(a.subset_seed).sample([r.rid for r in routes], a.n)); routes = [r for r in routes if r.rid in keep]
    sp = specs(a.ortools); out = Path(a.out); out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        d = pd.read_csv(out); cnt = d.groupby("route").method.nunique(); done = set(cnt[cnt >= len(sp)].index)
    todo = [r for r in routes if r.rid not in done]
    print(f"{len(routes)} routes, {len(done)} already done, {len(todo)} to do, {len(sp)} methods each, seed {a.seed}", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), a.chunk):
        part = todo[i:i + a.chunk]
        df = run_parallel(part, sp, a.workers, None, a.seed, progress=False)
        df.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"chunk {i // a.chunk + 1}: routes {i + len(part)}/{len(todo)}  elapsed {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(vars(a), specs=sp, n_routes=len(routes), wall_seconds=round(time.time() - t0), date=time.strftime("%Y-%m-%d")), open(out.with_suffix(".json"), "w"), indent=2)
    print("done", round(time.time() - t0), "s")
